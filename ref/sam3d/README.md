# ref/sam3d — pytorch-reference dumps for sam-3d-objects

Per-stage `.npy` tensors used by `cpu/sam3d/verify_*.c`,
`cuda/sam3d/verify_*.c` and `rdna4/sam3d/verify_*.c` to diff our C /
CUDA / HIP ports against the reference pytorch pipeline.

## TL;DR — generate on a CUDA host, copy to your AMD host

The upstream `sam3d_objects` package depends on `pytorch3d`, `xformers`,
`spconv-cu121`, `cuda-python`, etc. There is **no PyPI/conda
pytorch3d Linux wheel built against ROCm torch**, so this dump must be
produced on a CUDA-capable host. After the dumps land in
`/tmp/sam3d_ref/`, `rsync` them to the AMD machine and run the HIP
verifiers there.

```bash
# === ON A CUDA HOST (Linux + Python 3.11 + NVIDIA GPU + driver ≥ 525) ===
git clone <this repo> gemm && cd gemm/ref/sam3d
uv venv --python 3.11 .venv
VIRTUAL_ENV=$PWD/.venv uv pip install \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.ngc.nvidia.com \
    -e /path/to/sam-3d-objects   # see "Upstream package" below

# Pull the gated facebook/sam-3d-objects checkpoint to $MODELS/sam3d/
# (HF login required — visit the model page once and accept terms).
huggingface-cli download facebook/sam-3d-objects \
    --local-dir $MODELS/sam3d/checkpoints

source .venv/bin/activate
python gen_image_ref.py \
    --image  /tmp/sam-3d-objects/notebook/images/human_object/image.png \
    --mask   /tmp/sam-3d-objects/notebook/images/human_object/0.png \
    --pipeline-yaml $MODELS/sam3d/checkpoints/pipeline.yaml \
    --outdir /tmp/sam3d_ref \
    --seed 42 --steps 25 --slat-steps 25 --cfg 7.5

# === COPY TO YOUR AMD/ROCm HOST ===
rsync -av /tmp/sam3d_ref/ amd-host:/tmp/sam3d_ref/

# === ON THE AMD HOST: run the HIP verifiers ===
cd rdna4/sam3d
make verify-dinov2 verify-cond-fuser verify-ss-dit verify-ss-decoder \
     verify-slat-dit verify-slat-gs
```

## Why a CUDA host is required

`sam3d_objects.pipeline.inference_pipeline` imports `pytorch3d` (for
`Meshes`, `Transform3d`, `quaternion_to_matrix`, …). The only
pre-built `pytorch3d` Linux wheels on PyPI / conda-forge / Anaconda
are CUDA-built (`cu121`, `cu118`); they refuse to load against a ROCm
torch. Source-building `pytorch3d` against ROCm torch is possible but
out of scope here. Other CUDA-only deps in the stack: `spconv-cu121`,
`xformers`, `flash-attn`, `cuda-python`, `bitsandbytes`. The script
`gen_image_ref.py` itself is CUDA-agnostic — the constraint comes from
the upstream package it instantiates.

## Upstream package

Clone Meta's `sam-3d-objects` next to this repo (or anywhere; pass the
path to `pip install -e`). The `gen_image_ref.py` script needs the
package importable, not necessarily installed editable — a `.pth`
shim works too:

```bash
# Option A: editable install
uv pip install -e /path/to/sam-3d-objects

# Option B: just make it importable (skips upstream's gated bootstrap)
echo /path/to/sam-3d-objects \
    > .venv/lib/python3.11/site-packages/sam3d_objects.pth
```

The repo's `__init__.py` runs a Meta-internal `lidra.init` bootstrap
that isn't in the OSS release; `gen_image_ref.py` sets
`LIDRA_SKIP_INIT=1` before import to skip it.

## What to dump

`gen_image_ref.py` registers `forward_hook`s on every stage and writes
the output of each to `/tmp/sam3d_ref/<stage>.npy` as f32. It also
captures the canonical end-to-end `pipe.run(...)` outputs as
`out_<key>.npy`.

Stages dumped:

| File                 | Shape               | Dtype | Source stage                           |
|----------------------|---------------------|-------|----------------------------------------|
| `input_image.npy`    | (H, W, 3)           | u8    | PIL load (always written)              |
| `input_mask.npy`     | (H, W)              | u8    | PIL load (always written)              |
| `pointmap.npy`       | (H, W, 3)           | f32   | MoGe (or `--pointmap` override)        |
| `dinov2_tokens.npy`  | (T, 1024)           | f32   | DINOv2-L/14 + 4 reg tokens (post drop) |
| `cond_tokens.npy`    | (N, 1024)           | f32   | CondEmbedderFuser                      |
| `ss_latent.npy`      | (8, 16, 16, 16)     | f32   | SS Flow DiT → latent                   |
| `occupancy.npy`      | (64, 64, 64)        | f32   | SS-VAE decoder                         |
| `slat_feats.npy`     | (M, C)              | f32   | SLAT Flow DiT features                 |
| `slat_coords.npy`    | (M, 4)              | i32   | SLAT voxel coordinates                 |
| `gaussians.npy`      | (G, 17)             | f32   | SLAT GS decoder (PLY channel order)    |
| `out_*.npy`          | varies              | f32   | end-to-end `pipe.run()` return dict    |

## Per-stage I/O dumps

The verifiers under `rdna4/sam3d/verify_*_realw.c` (and their
`cpu/sam3d/cuda/sam3d` siblings) expect a richer set of intermediate
tensors than `gen_image_ref.py` captures. Those come from four
companion scripts that each instantiate one model wrapper directly,
side-stepping the full pipeline (so they do **not** need MoGe or
nvdiffrast):

| Script                | Produces under `/tmp/sam3d_ref/`                      |
|-----------------------|--------------------------------------------------------|
| `dump_cond_tokens.py` | `cond_tokens.npy`, `dinov2_tokens.npy`, …              |
| `dump_ss_dit_io.py`   | `ss_dit_{in,out}_*.npy`, `ss_dit_{cond,t,d}.npy`       |
| `dump_ss_dec_io.py`   | `ss_dec_in.npy`, `ss_dec_out.npy`                      |
| `dump_slat_dit_io.py` | `slat_dit_{in_coords,in_feats,t,cond,out_feats}.npy`   |
| `dump_slat_gs_io.py`  | `slat_gs_*.npy` (GS-decoder I/O)                       |

Run them in order after `gen_image_ref.py` (each picks up files the
previous step wrote — e.g. `dump_ss_dit_io.py` defaults to
`--cond-npy /tmp/sam3d_ref/cond_tokens.npy`):

```bash
python dump_cond_tokens.py  --pipeline-yaml $MODELS/sam3d/checkpoints/pipeline.yaml --outdir /tmp/sam3d_ref
python dump_ss_dit_io.py    --pipeline-yaml $MODELS/sam3d/checkpoints/pipeline.yaml --outdir /tmp/sam3d_ref
python dump_ss_dec_io.py    --pipeline-yaml $MODELS/sam3d/checkpoints/pipeline.yaml --outdir /tmp/sam3d_ref
python dump_slat_dit_io.py  --pipeline-yaml $MODELS/sam3d/checkpoints/pipeline.yaml --outdir /tmp/sam3d_ref
python dump_slat_gs_io.py   --pipeline-yaml $MODELS/sam3d/checkpoints/pipeline.yaml --outdir /tmp/sam3d_ref
```

These scripts set `SPCONV_ALGO=native` so they run on CPU when the
host's CUDA arch isn't covered by `spconv-cu121`'s shipped kernel
images — but they still need a ROCm-incompatible PyTorch / spconv to
import, so they belong on the same CUDA host as `gen_image_ref.py`.

## Determinism notes

* `torch.manual_seed` + `numpy.random.seed` set from `--seed`.
* `torch.set_float32_matmul_precision("highest")` to avoid TF32 drift.
* Model is forced to fp32 in the dump path (bf16 inference path is
  used in production; diff budgets for each `verify_` stage allow for
  the bf16 floor).

## Dump without the model

For step-1 smoke-testing the scaffold (before the HF checkpoint
arrives), `--skip-run` only dumps the preprocessed image and mask:

```bash
python gen_image_ref.py \
    --image /tmp/sam-3d-objects/notebook/images/human_object/image.png \
    --mask  /tmp/sam-3d-objects/notebook/images/human_object/0.png \
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

## ROCm host: what does work

If you only care about the end-to-end PLY (and not the per-stage
verifiers), `rdna4/sam3d/test_hip_sam3d` runs standalone — it needs
only the safetensors weights, no PyTorch ref dumps:

```bash
cd rdna4/sam3d
./test_hip_sam3d \
    --safetensors-dir /mnt/disk1/models/sam3d/safetensors \
    /tmp/sam-3d-objects/notebook/images/human_object/image.png \
    /tmp/sam-3d-objects/notebook/images/human_object/0.png \
    --steps 2 --slat-steps 12 --cfg 2.0 --precision fp32 \
    -o /tmp/hip_sam3d_e2e.ply
```

Likewise, the weights-only verifiers (`verify_ssdit_block_realw`,
`verify_ssdit_full_realw`) build their reference inputs from the
safetensors directly and skip `/tmp/sam3d_ref/` entirely. Use those
for a quick numerics gate when you don't have access to a CUDA host.

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
