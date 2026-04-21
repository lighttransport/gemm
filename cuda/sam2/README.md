# CUDA SAM2 (reference-backed runtime)

`cuda/sam2` now provides a C API + CLI that runs end-to-end promptable
SAM2 inference via the local `ref/sam2/gen_image_ref.py` pipeline, then returns
masks/scores through the C API.

Native CUDA stages currently integrated:
- image preprocess kernel (`sam2_preprocess`)
- point/box prompt rasterization kernels (`sam2_points_to_map`, `sam2_box_to_map`)
- mask threshold kernel (`sam2_threshold_u8`)
- image encoder patch embedding verification path (`verify_patch_embed`)
- image encoder block-0 verification path (`verify_block0`)

Core SAM2 backbone/decoder inference is still reference-backed while native
CUDA model kernels are being ported.

## Weights

Download SAM2.1 tiny/small checkpoints into `/mnt/disk01/sam2/`:

```bash
./scripts/download_sam2_weights.sh
```

## Build

```bash
cd cuda/sam2
make
```

Patch-embed verification (native CUDA vs PyTorch):

```bash
python3 ref/sam2/gen_patch_embed_ref.py \
  --model /mnt/disk01/sam2/sam2.1-hiera-tiny \
  --image /tmp/hy3d_textured.jpg \
  --outdir /tmp/sam2_patch_ref

./verify_patch_embed \
  /mnt/disk01/sam2/sam2.1-hiera-tiny/model.safetensors \
  /tmp/sam2_patch_ref
```

Block-0 trace verification (native CUDA vs PyTorch):

```bash
python3 ref/sam2/gen_block0_trace_ref.py \
  --model /mnt/disk01/sam2/sam2.1-hiera-tiny \
  --image /tmp/hy3d_textured.jpg \
  --outdir /tmp/sam2_b0_trace_ref

./verify_block0 \
  /mnt/disk01/sam2/sam2.1-hiera-tiny/model.safetensors \
  /tmp/sam2_b0_trace_ref
```

Current status for block-0 CUDA verifier:
- `ln1` is numerically tight.
- drift starts at `attn`, then amplifies in downstream `ln2/mlp`.

## Run

```bash
./test_cuda_sam2 /mnt/disk01/sam2/sam2.1-hiera-tiny /path/to/image.jpg \
  --point 256 256 1 -o /tmp/sam2_masks.npy
```

Multiple prompts:

```bash
./test_cuda_sam2 /mnt/disk01/sam2/sam2.1-hiera-small /path/to/image.jpg \
  --point 220 280 1 --point 300 320 1 --box 150 180 420 460
```

Prerequisites for runtime path:
- `python3`
- `transformers`, `torch`, `pillow`, `numpy`
- CUDA driver + NVRTC runtime (`libcuda.so`, `libnvrtc.so`) for native CUDA stages
