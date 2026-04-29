# ref/sam3d-body — pytorch-reference dumps for SAM 3D Body

Per-stage `.npy` tensors used by `cpu/sam3d_body/verify_*.c` (and later
`cuda/sam3d_body/verify_*.c`) to diff our C/CUDA ports against the
upstream `facebookresearch/sam-3d-body` pipeline.

## Python environment

The dump script runs the upstream `sam_3d_body` package (pinned to
`torch` + `pytorch-lightning` + `detectron2@a1ce2f9` per upstream
INSTALL.md). A dedicated `uv` venv lives at `ref/sam3d-body/.venv`.

```bash
cd ref/sam3d-body
uv venv --python 3.11 .venv

# Upstream package (editable install from cloned source):
test -d /tmp/sam-3d-body || git clone --depth 1 \
    https://github.com/facebookresearch/sam-3d-body /tmp/sam-3d-body
VIRTUAL_ENV=$PWD/.venv uv pip install \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    -e /tmp/sam-3d-body

# Deps from upstream INSTALL.md (single flat list):
VIRTUAL_ENV=$PWD/.venv uv pip install \
    torch torchvision \
    pytorch-lightning pyrender opencv-python yacs scikit-image einops \
    timm dill pandas rich hydra-core pyrootutils webdataset chump \
    "networkx==3.2.1" roma joblib seaborn appdirs fvcore tensorboard \
    huggingface_hub loguru optree jsonlines pycocotools xtcocotools

# detectron2 (needed only when re-enabling auto-crop; v1 can skip):
# VIRTUAL_ENV=$PWD/.venv uv pip install \
#     "git+https://github.com/facebookresearch/detectron2.git@a1ce2f9" \
#     --no-build-isolation --no-deps
```

`--index-strategy unsafe-best-match` lets cuda wheels from
pytorch.org mix with PyPI. The gated HF checkpoints need one-time
auth:

```bash
source .venv/bin/activate
hf auth login   # or: huggingface-cli login (legacy)
```

## Gated model download

```bash
export MODELS=/mnt/disk01/models
hf download facebook/sam-3d-body-dinov3 --local-dir $MODELS/sam3d-body/dinov3
hf download facebook/sam-3d-body-vith   --local-dir $MODELS/sam3d-body/vith
```

Both repos ship `model.ckpt` (encoder+decoder+head weights) and
`assets/mhr_model.pt` (MHR parametric-body assets). The ViT-H variant
is downloaded for convenience but not consumed by v1 (DINOv3 path only).

## One-shot dump

```bash
source ref/sam3d-body/.venv/bin/activate
export MODELS=/mnt/disk01/models

python ref/sam3d-body/gen_image_ref.py \
    --image person.jpg \
    --hf-repo-id facebook/sam-3d-body-dinov3 \
    --outdir /tmp/sam3d_body_ref --seed 42
```

The script registers `forward_hook`s on the encoder, decoder, MHR
head, camera head, and MHR skinning step; each captured tensor is
written to `/tmp/sam3d_body_ref/<stage>.npy` as `f32`.

## Files produced

| File                       | Shape              | Dtype | Source stage                       |
|----------------------------|--------------------|-------|------------------------------------|
| `input_image.npy`          | (H, W, 3)          | u8    | raw RGB (always written)           |
| `image_processed.npy`      | (Hc, Wc, 3)        | f32   | crop+resize+normalize              |
| `dinov3_tokens.npy`        | (N, D)             | f32   | DINOv3 encoder output              |
| `mhr_params.npy`           | (519,)             | f32   | MHR head regressor output          |
| `cam_params.npy`           | (4,)               | f32   | [cam_t_x, cam_t_y, cam_t_z, focal] |
| `out_vertices.npy`         | (V, 3)             | f32   | MHR-skinned vertices in cam frame  |
| `out_keypoints_3d.npy`     | (K, 3)             | f32   | 3D joints in cam frame             |
| `out_keypoints_2d.npy`     | (K, 2)             | f32   | projected pixels                   |
| `out_faces.npy`            | (F, 3)             | i32   | MHR triangle indices (const)       |

## Determinism

- `torch.manual_seed(args.seed) + np.random.seed(args.seed)`
- `torch.set_float32_matmul_precision("highest")` to avoid TF32 drift
- Encoder forced fp32 for ref dumps (the production path uses bf16
  on CUDA; our diff tolerances account for the bf16 floor ≈ 1e-2)

## Dump without the model (smoke)

For step-0 scaffolding before the full pipeline installs:

```bash
python ref/sam3d-body/gen_image_ref.py \
    --image person.jpg --outdir /tmp/sam3d_body_ref --skip-run
```

Only `input_image.npy` and `image_processed.npy` are written — enough
to exercise the NPY reader in the verifiers.

## Deferred

- Detectron2-based person detection / auto-crop — user supplies bbox
  (or uses the full frame). Add when the encoder chain is green.
- Optional 2D-keypoint / mask prompts — upstream supports them but v1
  runs RGB-only.
- ViT-H variant — separate dump pass once DINOv3 variant is end-to-end
  green.
