# ref/sam3d — pytorch-reference dumps for sam-3d-objects

Per-stage .npy tensors used by `cpu/sam3d/verify_*.c` (and later
`cuda/sam3d/verify_*.c`) to diff our C/CUDA ports against the
reference pytorch pipeline.

## Python environment

This ref dump needs the upstream `sam3d_objects` package (which pins
`torch==2.5.1+cu121`, `xformers==0.0.28.post3`, `spconv-cu121`, etc.).
A dedicated `uv` venv lives at `ref/sam3d/.venv`:

```bash
cd ref/sam3d
uv venv --python 3.11 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.ngc.nvidia.com \
    -e /path/to/sam-3d-objects
```

`--index-strategy unsafe-best-match` is required because `cuda-python`
and `spconv-cu121` live on NVIDIA's NGC index, not on PyPI or
pytorch.org's index. The `+cu121` torch wheels are forward-compatible
with newer host drivers (host CUDA 13.x + driver 570+ both fine).

## One-shot dump

```bash
source ref/sam3d/.venv/bin/activate      # activate the venv above

# HF checkpoint is gated (facebook/sam-3d-objects); download manually.
export MODELS=/mnt/disk01/models

python ref/sam3d/gen_image_ref.py \
    --image  fujisan.jpg \
    --mask   fujisan_mask.png \
    --pipeline-yaml $MODELS/sam3d/pipeline.yaml \
    --outdir /tmp/sam3d_ref \
    --seed 42 --steps 25 --slat-steps 25 --cfg 7.5
```

The script registers `forward_hook`s on every stage and writes the
output of each to `/tmp/sam3d_ref/<stage>.npy` as f32.

## Files produced

| File                 | Shape               | Dtype | Source stage                           |
|----------------------|---------------------|-------|----------------------------------------|
| `input_image.npy`    | (H, W, 3)           | u8    | PIL load (always written)              |
| `input_mask.npy`     | (H, W)              | u8    | PIL load (always written)              |
| `pointmap.npy`       | (H, W, 3)           | f32   | MoGe or user-supplied pointmap         |
| `dinov2_tokens.npy`  | (257, 768)          | f32   | DINOv2-B/14 @ 224                      |
| `cond_tokens.npy`    | (N, C)              | f32   | CondEmbedderFuser                      |
| `ss_latent.npy`      | (8, 16, 16, 16)     | f32   | SS Flow DiT → latent                   |
| `occupancy.npy`      | (64, 64, 64)        | f32   | SS-VAE decoder                         |
| `slat_feats.npy`     | (M, C)              | f32   | SLAT Flow DiT features                 |
| `slat_coords.npy`    | (M, 4)              | i32   | SLAT voxel coordinates                 |
| `gaussians.npy`      | (G, 17)             | f32   | SLAT GS decoder (PLY channel order)    |

## Determinism notes

* `torch.manual_seed` + `numpy.random.seed` set from `--seed`.
* `torch.set_float32_matmul_precision("highest")` to avoid TF32 drift.
* Model is forced to fp32 in the dump path (bf16 inference path is
  used in production; diff budgets for each verify_ stage allow for
  the bf16 floor).

## Dump without the model

For step-1 smoke-testing the scaffold (before the HF checkpoint
arrives), use `--skip-run` to only dump the preprocessed image and
mask:

```bash
python ref/sam3d/gen_image_ref.py \
    --image fujisan.jpg --mask fujisan_mask.png \
    --pipeline-yaml /dev/null \
    --outdir /tmp/sam3d_ref --skip-run
```

## MoGe pointmaps

`moge_pointmap.py` runs the installed MoGe package and writes the
`(H, W, 3)` float32 pointmap consumed by the C/CUDA runners:

```bash
ref/sam3d/.venv/bin/python ref/sam3d/moge_pointmap.py \
    --image fujisan.jpg \
    --out /tmp/sam3d_ref/pointmap.npy
```

The script defaults to `/mnt/disk01/models/moge-vitl/model.pt` when it
exists, otherwise to the HF id `Ruicheng/moge-vitl`. The CUDA CLI can
invoke this helper directly with `--moge`; `--moge-out p.npy` preserves
the generated pointmap instead of using a temporary file.

## Deferred

* SLAT mesh decoder: native SDF/deformation/vertex-RGB and
  FlexiCubes-style beta/alpha/gamma topology extraction are wired in the
  CUDA CLI via `--mesh-source slat`; upstream ambiguity/check-table
  parity is implemented. `--mesh-texture-size N` unwraps with vendored
  xatlas and writes an embedded PNG textured GLB; decoder RGB is the
  fallback color source, while `--mesh-texture-color image` uses the
  input image plus finite masked pointmap for deterministic single-view
  source projection. `--mesh-texture-mode grid` forces the legacy
  duplicated triangle-grid atlas for comparison; in that mode
  `--mesh-texture-size` is a minimum and may grow to preserve 4x4 texel
  tiles per triangle. High-quality multiview source/appearance
  optimization remains out of v1.
