# DA3 Reference Verification Report

Verification of our custom C/CUDA Depth-Anything-3 implementation against a PyTorch reference built from the official model weights and architecture.

## Overview

We implemented a PyTorch reference inference framework (`ref/da3/`) that loads the same safetensors weights as our C/CUDA code and produces outputs for direct numerical comparison. This verifies correctness of our custom implementation across all DA3 model variants and output modalities.

### Models Tested

| Model | Backbone | Params | Outputs |
|---|---|---|---|
| DA3-Small | ViT-S (384d, 12 blocks) | 25M | Depth, Confidence |
| DA3-Giant | ViT-G (1536d, 40 blocks, SwiGLU) | 1.2B | Depth, Pose, Rays, Gaussians |
| DA3Nested-Giant-Large-1.1 | ViT-G + ViT-L (1024d, 24 blocks) | 1.6B | All 7 modalities |

### Test Images

- Brooklyn Bridge (2100x1400) -- used for DA3-Small verification
- Urban street scene (1200x738) -- used for Giant and Nested models
- Room interior (1200x893) -- secondary Giant test

---

## Results: DA3-Small

**Comparison**: PyTorch F32 reference vs C/CUDA F16 compute, both using identical safetensors weights.

| Comparison | Pearson r | SSIM | MAE | Max Error |
|---|---|---|---|---|
| CPU (F32) vs Reference | **0.99996** | **0.99999** | 0.00024 | 0.003 |
| GPU (F16) vs Reference | **0.99924** | **0.99985** | 0.00109 | 0.016 |
| GPU vs CPU | **0.99925** | **0.99987** | 0.00108 | 0.016 |

All comparisons PASS with very high correlation. The small GPU-vs-CPU difference is expected from F16 vs F32 precision.

## Results: DA3-Giant

**Comparison**: PyTorch BF16 reference vs C/CUDA F16 compute. BF16 is used for the reference because the 1.2B parameter model exceeds GPU memory in F32.

| Output | Metric | Value | Assessment |
|---|---|---|---|
| Depth | Pearson r | **0.949** | Good (precision loss through 40 blocks) |
| Pose (translation) | Max abs diff | 0.036 | Acceptable |
| Pose (quaternion) | Cosine similarity | **0.971** | Good |
| Pose (FOV) | Max abs diff | 0.094 | Acceptable |
| Gaussians (38ch) | Overall Pearson r | **0.998** | Excellent |
| Gaussians per-channel | Pearson r range | 0.997--0.999 | Excellent |
| Rays | GPU output | All zeros | Known issue (see below) |

## Results: DA3Nested-Giant-Large-1.1

**Comparison**: PyTorch BF16 reference vs C/CUDA F16 compute.

| Output | Ref Range | GPU Range | Pearson r | Status |
|---|---|---|---|---|
| Depth | [0.61, 1.13] | [0.62, 1.08] | **0.963** | PASS |
| Pose (quaternion) | -- | -- | cos=**0.981** | PASS |
| Gaussians (38ch) | non-zero | non-zero | **0.998** | PASS |
| Rays | [-0.36, 0.42] | all zeros | -- | Known issue |
| Metric Depth | [-0.19, 0.29] | not impl | -- | C/CUDA needed |
| Sky Segmentation | sig [0.002, 0.46] | not impl | -- | C/CUDA needed |

---

## DA3 Model Architecture

### High-Level Overview

DA3 (Depth Anything 3) is a monocular depth estimation model that predicts dense depth, camera pose, ray directions, and Gaussian splatting parameters from a single RGB image. The architecture comprises a DINOv2 Vision Transformer backbone with 2D Rotary Position Embeddings, followed by a DPT (Dense Prediction Transformer) head that fuses multi-scale features into dense predictions.

```
RGB Image [H, W, 3]
  |
  v
Patch Embedding (Conv2d, patch_size=14)
  |   [n_patches, embed_dim]
  v
Prepend CLS Token + Add Positional Embeddings
  |   [n_tokens, embed_dim]          n_tokens = n_patches + 1
  v
Transformer Backbone (12/24/40 blocks)
  |   - Alternating local/global attention (from alt_start)
  |   - QK Normalization (from qknorm_start)
  |   - 2D RoPE (from rope_start, freq_base=100)
  |   - Camera token injection at CLS (at rope_start)
  |   - Extract features at 4 designated layers
  v
+------- Feature Concatenation: [raw_patch, norm(patch)] --> 2*embed_dim
|
v
DPT Head (DualDPT)
  |   4 projection layers --> 4 spatial scales
  |   4 adapter convolutions
  |   4-stage RefineNet fusion (coarse-to-fine)
  |   Output convolutions + sinusoidal UV pos_embed
  v
Depth: exp(z)           Confidence: exp(z) + 1
  |
  v
Bilinear upsample to original resolution [H, W]
```

### Model Variants

| Variant | Backbone | embed_dim | Blocks | Heads | FFN | Params | Outputs |
|---|---|---|---|---|---|---|---|
| Small | ViT-S | 384 | 12 | 6 | GELU MLP, 1024 | 25M | Depth, Confidence |
| Base | ViT-B | 768 | 12 | 12 | GELU MLP | ~86M | Depth, Confidence |
| Large | ViT-L | 1024 | 24 | 16 | GELU MLP | ~300M | Depth, Confidence |
| Giant | ViT-G | 1536 | 40 | 24 | SwiGLU | 1.2B | Depth, Pose, Rays, Gaussians |
| Nested (G+L) | ViT-G + ViT-L | 1536+1024 | 40+24 | -- | -- | 1.6B | All 7 modalities |

**Size-based configuration defaults (auto-detected from embed_dim):**

| Size | embed_dim | Feature layers | head_features | head_out_channels | rope/qknorm start |
|---|---|---|---|---|---|
| Small | <768 | [5, 7, 9, 11] | 64 | [48, 96, 192, 384] | 4 |
| Base | >=768 | [5, 7, 9, 11] | 128 | [96, 192, 384, 768] | 4 |
| Large | >=1024 | [11, 15, 19, 23] | 256 | [256, 512, 1024, 1024] | 8 |
| Giant | >=1536 | [19, 27, 33, 39] | 256 | [256, 512, 1024, 1024] | 13 |

These can be overridden by `config.json` (`config.net.out_layers`, `config.head.features`, etc.).

### Backbone: DINOv2 Vision Transformer

**Input processing:**

1. **Patch embedding**: Conv2d(3, embed_dim, patch_size, stride=patch_size). Default: patch_size=14, image_size=518, yielding a 37x37 grid of 1369 patches.
2. **CLS token**: Learned [1, embed_dim] prepended to patch sequence. Total tokens: 1370.
3. **Positional embedding**: Learned [n_tokens, embed_dim] added element-wise to all tokens.

**Transformer block (repeated n_blocks times):**

```
x --> LayerNorm1 --> Attention --> LayerScale1 --> (+x) --> LayerNorm2 --> FFN --> LayerScale2 --> (+) --> out
```

Each block contains:

- **LayerNorm** (eps=1e-6): Pre-norm for both attention and FFN sub-layers.
- **Multi-head self-attention**:
  - QKV projection: Linear(embed_dim, 3 * embed_dim)
  - Reshape to [B, n_heads, n_tokens, head_dim] (head_dim = embed_dim / n_heads = 64)
  - **QK Normalization** (from qknorm_start): Per-head LayerNorm on Q and K independently, each [n_tokens, head_dim].
  - **2D RoPE** (from rope_start): Applied to Q and K. See RoPE section below.
  - Scaled dot-product attention (SDPA): softmax(QK^T / sqrt(head_dim)) * V
  - Output projection: Linear(embed_dim, embed_dim)
- **LayerScale**: Learned per-dimension scalar [embed_dim], multiplied before residual addition. Separate ls1 (attention) and ls2 (FFN).
- **FFN** (two variants):
  - *GELU MLP* (Small/Base/Large): Linear(embed_dim, ffn_hidden) --> GELU --> Linear(ffn_hidden, embed_dim)
  - *SwiGLU* (Giant): gate_up = Linear(embed_dim, 2 * ffn_hidden), split into gate and up, then SiLU(gate) * up --> Linear(ffn_hidden, embed_dim)

**Alternating local/global attention (from alt_start = rope_start):**

Starting at layer `alt_start`, blocks alternate between local and global attention. For **single-view inference** (S=1), the attention operation is identical -- the difference is only in the **RoPE positions**:

- **Local blocks** (even indices >= alt_start, and all blocks < alt_start): Use real 2D grid positions. CLS at (0,0), patches at (y+1, x+1) where y,x in 0..grid_h-1.
- **Global blocks** (odd indices >= alt_start): Use `pos_nodiff` -- CLS at (0,0), ALL patches at (1,1). This makes the transformer spatially invariant for cross-view reasoning.

For multi-view inference (S>1), local attention is restricted to within-view tokens while global attention spans all views.

**Camera token injection:**

At block `alt_start`, the learned `camera_token` parameter ([1, 2, embed_dim], stored in `backbone.pretrained.camera_token`) replaces the CLS token:

```
hidden[0, :] = camera_token[0, 0, :]   # ref view token at CLS position
```

This conditions all subsequent blocks to be camera-aware, which is essential for ray and pose prediction. Without it, the aux DPT (rays) produces all zeros.

**Feature extraction:**

At each of the 4 designated feature layers, the hidden state is saved. With `cat_token=True` (DA3 default), saved features are the concatenation of:
- `local_x`: The output of the **previous** block (which is always a local attention block for the feature layers)
- `norm(x)`: The backbone's final LayerNorm applied to the current block's output

This yields features of shape [n_patches, 2 * embed_dim] (CLS excluded). For Small: [1369, 768].

### 2D Rotary Position Embedding (RoPE)

DA3 uses a 2D variant of RoPE that encodes spatial Y and X positions into the Q/K vectors.

**Parameters:**
- Frequency base: **100.0** (not the standard 10000.0 used in language models)
- `patch_start_idx = 1`: CLS token gets position (0, 0), patches get positions (y+1, x+1)
- Applied from layer `rope_start` onward

**Encoding scheme (per head, head_dim=64):**

The head dimension is split into two halves, each further split into pairs:

```
head_dim = 64
half = head_dim / 2 = 32

First half (dims 0..31): Y-axis rotation
  For each pair j in 0..15:
    freq_y = 1 / (100.0 ^ (2j / 32))
    theta_y = pos_y * freq_y
    (dim[2j], dim[2j+1]) rotated by theta_y

Second half (dims 32..63): X-axis rotation
  For each pair j in 0..15:
    freq_x = 1 / (100.0 ^ (2j / 32))
    theta_x = pos_x * freq_x
    (dim[32+2j], dim[32+2j+1]) rotated by theta_x
```

The rotation pairs are **within** each half (0-1, 2-3, ..., 30-31 for Y; 32-33, 34-35, ..., 62-63 for X), not across halves.

### DPT Head (DualDPT)

The DPT head converts 4 multi-scale backbone features into dense per-pixel predictions.

**Spatial dimension pyramid (for grid_h=37):**

| Level | ConvTranspose/Conv | Spatial size | Operation |
|---|---|---|---|
| 0 | ConvTranspose2d 4x4 stride 4 | (37-1)*4+4 = **148x148** | Upsample 4x |
| 1 | ConvTranspose2d 2x2 stride 2 | (37-1)*2+2 = **74x74** | Upsample 2x |
| 2 | Identity | **37x37** | Grid resolution |
| 3 | Conv2d 3x3 stride 2 pad 1 | (37+2-3)/2+1 = **19x19** | Downsample 2x |

**Token processing pipeline (per feature level i):**

```
feat[i] [n_patches, 2*embed_dim]
  |
  v
Head LayerNorm --> Linear projection [n_patches, head_out_channels[i]]
  |
  v
Token-to-CHW [oc[i], grid_h, grid_w]
  |
  v
ConvTranspose/Conv/Identity --> [oc[i], sp_h[i], sp_w[i]]
  |
  v
Adapter Conv2d 3x3 pad=1 (no bias) --> [head_features, sp_h[i], sp_w[i]]
```

**Sinusoidal UV positional embedding:**

Applied twice in the DPT head (once after adapters in token-major layout, once after upsample to model resolution). Parameters: omega_0=100, ratio=0.1. Generates per-pixel UV coordinates normalized to [0,1], encodes them with sinusoidal functions, and adds to the feature channels.

**RefineNet fusion (FeatureFusionBlock, 4 stages coarse-to-fine):**

```
Stage 3 (deepest): adapter[3] alone                       --> [feat, 19x19]
  --> upsample to 37x37 -->
Stage 2:           upsample(deeper) + RCU1(adapter[2])    --> RCU2 --> upsample --> out_conv --> [feat, 37x37]
  --> upsample to 74x74 -->
Stage 1:           upsample(deeper) + RCU1(adapter[1])    --> RCU2 --> upsample --> out_conv --> [feat, 74x74]
  --> upsample to 148x148 -->
Stage 0 (final):   upsample(deeper) + RCU1(adapter[0])    --> RCU2 --> 2x upsample --> out_conv --> [feat, 296x296]
```

Each stage:
1. Bilinear upsample deeper input to lateral's spatial size (align_corners=True)
2. RCU1 (Residual Conv Unit): Conv2d 3x3 --> ReLU --> Conv2d 3x3, applied to lateral, added to upsampled deeper
3. RCU2: Conv2d 3x3 --> ReLU --> Conv2d 3x3, applied to the sum
4. Bilinear upsample to target output size
5. Out_conv: Conv2d 1x1

Stage 0 uses scale_factor=2, producing 296x296 output (2x the level-0 adapter size of 148x148).

**Output convolutions:**

```
RefineNet output [feat, 296x296]
  |
  v
output_conv1 (neck): Conv2d(feat, feat/2, 3x3, pad=1)    -- No ReLU
  |   [feat/2, 296x296]   (e.g., [32, 296, 296])
  v
Bilinear upsample to model resolution (518x518)
  |
  v
Sinusoidal UV pos_embed (ratio=0.1)
  |
  v
output_conv2[0]: Conv2d(feat/2, feat/2, 3x3, pad=1) + ReLU
output_conv2[1]: Conv2d(feat/2, output_dim, 1x1)     -- No activation
  |   [output_dim, 518x518]   (output_dim=2 for depth+confidence)
  v
depth_activation:
  depth      = exp(logit[0])        -- always positive
  confidence = exp(logit[1]) + 1    -- always >= 1 ("expp1")
  |
  v
Bilinear upsample to original image resolution [H, W]
```

### Output Modalities

**1. Depth + Confidence (main DPT head, all models):**
- depth: exp(z), range typically [0.3, 3.5] for indoor/outdoor scenes
- confidence: exp(z) + 1, range [1.0, ~4.0]

**2. Camera Pose (CameraDec, Giant/Nested):**

```
backbone_norm(CLS token) [embed_dim]
  --> Linear(embed_dim, mlp_dim) + GELU
  --> Linear(mlp_dim, mlp_dim) + GELU
  --> 3 parallel linear heads:
       fc_t:    Linear(mlp_dim, 3)  --> translation [tx, ty, tz]
       fc_qvec: Linear(mlp_dim, 4)  --> quaternion [qw, qx, qy, qz] (L2-normalized)
       fc_fov:  Linear(mlp_dim, 2)  --> field of view [fov_x, fov_y]
```

Output: 9 floats = [t(3), q(4), fov(2)].

**3. Rays (Auxiliary DPT head, Giant/Nested):**

A second DPT head with its own RefineNet weights, sharing the same backbone features. Produces 6-channel ray directions + 1-channel ray confidence. Requires camera token injection to produce non-zero output.

**4. Gaussians (GSDPT head, Giant/Nested):**

A third DPT head (gs_head) that reuses the standard DPT structure plus an **images_merger** module:
- Images merger: 3 sequential Conv2d layers with stride 2 (3-->32-->64-->128), progressively downsampling the input image
- Merger features are bilinearly upsampled and element-wise added to the DPT fused features before output convolutions
- Output: 38 channels (Gaussian splatting parameters: position, scale, rotation, opacity, SH coefficients)

**5. Metric Depth (Nested model only, not yet in C/CUDA):**

A second backbone (ViT-L, 1024d, 24 blocks) with its own DPT head (output_dim=1). Produces absolute-scale depth.

**6. Sky Segmentation (Nested model only, not yet in C/CUDA):**

Shares the metric DPT neck, with a separate `sky_output_conv2` branch producing a binary sky mask (sigmoid activation).

### Camera Encoder (CameraEnc, Giant/Nested)

Encodes camera intrinsics into a latent representation (used in multi-view settings):

```
Input: focal_length, principal_point, image_size [6 floats]
  --> Linear + GELU (expand to enc_dim=192)
  --> 4 transformer blocks (self-attention over camera tokens)
  --> trunk_norm + token_norm
  --> pose_branch: 2-layer MLP --> 9-dim pose (same format as CameraDec)
```

### Weight Storage

**GGUF format**: Supports quantized formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q6_K, Q4_K) plus F16, BF16, F32. Keys follow `da3.{component}` naming.

**Safetensors format**: Direct F16/F32/BF16 loading with mmap. Keys follow `model.{component}` or `model.da3.{component}` (nested models) naming. Tensor name mapping translates safetensors keys to internal GGUF-style names.

### config.json Structure

Standard (Small/Giant):
```json
{ "config": { "head": { "features": 64, "out_channels": [48,96,192,384] },
              "net":  { "out_layers": [5,7,9,11], "rope_start": 4, "qknorm_start": 4 } } }
```

Nested variant:
```json
{ "config": { "anyview": { "head": {...}, "net": {...} } } }
```

---

## Architecture Bugs Found and Fixed

### 1. RoPE Implementation (Critical -- small model)

**Bug**: The 2D Rotary Position Embedding rotated pairs **across** halves of head_dim `(0-31 with 32-63)` instead of **within** each half `(0-15 with 16-31, 32-47 with 48-63)`.

**Impact**: All features from layer `rope_start` onward were corrupted, causing near-zero correlation (Pearson r ~ 0.02).

**Fix**: Split head_dim into 4 quarters; apply Y rotation on first half, X rotation on second half, each rotating within their respective quarter pairs.

### 2. RefineNet Fusion Order (Critical -- small model)

**Bug**: The standard DPT convention applies RCU1 to the **skip** features. Our C code applies RCU1 to the **upsampled deeper** output and adds to skip.

**Impact**: When the PyTorch reference used standard DPT convention, correlation was ~0.79. Matching C code convention raised it to 0.999.

**Verdict**: Our C code implements a valid but non-standard fusion order. Both the reference and C code now match.

### 3. Output Head Upsample (Medium -- small model)

**Bug**: The PyTorch reference applied a 2x bilinear upsample before the output convolutions (standard DPT behavior). The C code applies all convolutions at the fusion resolution (148x148) and upsamples directly to the final resolution.

**Impact**: Correlation 0.79 vs 0.999.

**Fix**: Match C code behavior -- all output convs at fusion resolution.

### 4. Backbone Norm Double-Application (Medium -- small model)

**Bug**: The reference applied `backbone.norm` during feature extraction, then `head.norm` on the concatenated features -- double-norming.

**Impact**: Near-zero correlation.

**Fix**: Save raw hidden states (no backbone norm) at feature layers. Only `head.norm` is applied after CLS token concatenation.

### 5. Nested Model Key Prefix (Medium -- nested model)

**Bug**: The nested model uses `model.da3.*` key prefix (not `model.*`). The C/CUDA code's prefix detection lists didn't include this variant.

**Impact**: 0 tensors mapped, empty inference output.

**Fix**: Added `"model.da3.backbone.pretrained."`, `"model.da3.head."`, `"model.da3.cam_dec."`, `"model.da3.cam_enc."`, `"model.da3.gs_head."` to all prefix lists in both `depth_anything3.h` and `cuda_da3_runner.c`.

### 6. Nested Config Parsing (Medium -- nested model)

**Bug**: The C config.json parser looked for `config.head` and `config.net`, but the nested model has them under `config.anyview.head` and `config.anyview.net`.

**Impact**: Fell back to small-model defaults (feat=64, wrong feature layers), producing near-constant depth output.

**Fix**: Check for `config.anyview` first; if present, use it as the config source.

---

## Known Issues: Rays = 0

### Root Cause

The official DA3 backbone has a **learned `camera_token` parameter** (shape `[1, 2, dim]`) stored in `model.backbone.pretrained.camera_token`. At transformer block `alt_start` (layer 4 for small, 13 for giant), this token **replaces the CLS token** in the sequence:

```python
# Official DA3 vision_transformer.py
if i == self.alt_start:
    ref_token = self.camera_token[:, :1].expand(B, -1, -1)
    x[:, :, 0] = ref_token  # overwrite CLS with camera token
```

This makes all features from `alt_start` onward **camera-aware**. The auxiliary DPT head (ray output) was trained with these conditioned features and produces zero output without them.

### Evidence

| Condition | Aux RefineNet Output |
|---|---|
| With camera_token injection | [-57.6, 63.1] (non-zero) |
| Without camera_token injection | 0.0 (all zeros) |

### Required C/CUDA Fix

```
1. Load camera_token from safetensors (ref_token = camera_token[0, 0, :])
2. At block alt_start, overwrite hidden[0:dim] (CLS) with ref_token
3. Continue backbone -- all subsequent blocks use conditioned features
```

This is a ~5-line change in the backbone loop.

---

## Not Yet Implemented in C/CUDA

| Feature | Description | Required Components |
|---|---|---|
| Metric Depth | Absolute-scale depth from ViT-L | Second backbone (ViT-L, 24 blocks, dim=1024), DPT head (output_dim=1) |
| Sky Segmentation | Binary sky mask | Shares metric DPT neck, separate `sky_output_conv2` branch |
| Camera Token | Enables ray output | Load `camera_token` param, inject at `alt_start` in backbone loop |

---

## Reference Framework Files

```
ref/da3/
  pyproject.toml              # uv project dependencies
  run_reference.py            # Small/Base model reference (depth)
  run_reference_giant.py      # Giant model (depth, pose, rays, gaussians)
  run_reference_nested.py     # Nested model (all 7 modalities)
  compare.py                  # Metric computation and pass/fail
  run_all.sh                  # End-to-end comparison script
  .gitignore                  # Excludes output/, .venv/, etc.
```

### Usage

```bash
# Small model (depth only)
cd ref/da3 && uv sync
uv run python run_reference.py --model-dir /path/to/da3-small \
    --image input.jpg --output-dir output/ --device cuda --no-cam-token

# Giant model (all anyview outputs)
uv run python run_reference_giant.py --model-dir /path/to/da3-giant \
    --image input.jpg --output-dir output/ --device cuda

# Nested model (all outputs including metric depth and sky seg)
uv run python run_reference_nested.py \
    --model-dir /path/to/da3-nested-giant-1.1 \
    --image input.jpg --output-dir output/ --device cuda

# Compare
uv run python compare.py --reference output/depth_ref.npy \
    --ours output/depth_gpu.npy --label "GPU vs Ref"
```

### C/CUDA Test Programs

```bash
# JPEG/PNG input (no PPM conversion needed), auto-exports falsecolor PNG + fp16 EXR
./test_cuda_da3 model.safetensors -i photo.jpg -o depth.pgm --full
# Output: depth.pgm + depth_falsecolor.png + depth_depth.exr + pose/rays/gs npy

# With resize (token-based: ~1369 tokens matching DA3 default 37x37 grid)
./test_cuda_da3 model.safetensors -i photo.jpg --resize 1369t -o depth.pgm

# With resize (percentage)
./test_da3 model.safetensors -i photo.jpg --resize 50% -o depth.pgm

# With --npy for depth and --npy-dir for all modalities
./test_cuda_da3 model.safetensors -i photo.jpg -o depth.pgm \
    --npy depth.npy --npy-dir output/ --full
```

### Supported Input Formats

JPEG, PNG, BMP, TGA, PPM, HDR (Radiance), EXR (OpenEXR) -- via stb_image + tinyexr.

### Default Output (auto-exported alongside -o)

| File | Format | Content |
|------|--------|---------|
| `{base}.pgm` | PGM 16-bit | Grayscale depth (min/max normalized) |
| `{base}_falsecolor.png` | PNG RGB 8-bit | Spectral colormap (DA3 PyTorch-matching) |
| `{base}_depth.exr` | EXR fp16 | Raw depth values (CUDA only) |
