# TRELLIS.2 CUDA reference dumps — verification

This directory holds **ground-truth intermediate tensors** captured from the
CUDA reference path (`gen_image_to_3d.py` / `dump_ground_truth.py`) for
`pipeline_type='512'`, `seed=42`, conditioning image
`cpu/trellis2/trellis2_repo/assets/example_image/T.png` (alpha-synthesized).

The ROCm/HIP texgen path produces broken textures. The dumps here let you
bisect *where* that path diverges from the CUDA reference.

## Regenerating

```bash
cd cuda/trellis2
./run_dump_ground_truth.sh                   # stage boundaries only
DUMP_PER_STEP=1 ./run_dump_ground_truth.sh   # also per-diffusion-step latents
```

Override env vars: `IMAGE`, `MODEL_ROOT`, `DINOV3`, `OUTDIR`, `SEED`.

Toolchain at capture time (recorded into `manifest.json`):

- GPU: NVIDIA GeForce RTX 5060 Ti (sm_120, 16 GB)
- torch 2.7.1+cu128, triton 3.3.1, nvcc 12.9
- flash-attn 2.8.3 (sparse-attn backend), xformers 0.0.31.post1
- DINOv3 ViT-L/16 weights from `/mnt/disk01/models/dinov3-vitl16/`
- TRELLIS.2-4B checkpoints from `/mnt/disk01/models/trellis2-4b/`

## Pipeline boundaries dumped

The dumper mirrors `Trellis2ImageTo3DPipeline.run()` for `pipeline_type='512'`
and writes one `.npy` at every stage boundary. Suffix `.coords` / `.feats`
indicates a `SparseTensor` split into its int coordinate table and float
feature table.

| # | File | What | Shape | dtype |
|---|---|---|---|---|
| 0 | `00_image_preprocessed.npy` | RGB image after `preprocess_image()` (BiRefNet skipped, alpha was real) | `[H,W,3]` | uint8 |
| 1 | `01_dinov3_cond_512.npy` | DINOv3 features at image_size=512 (positive cond) | `[1,N,1024]` | bf16 |
| 1 | `01_dinov3_neg_cond_512.npy` | Zero negative cond, same shape | `[1,N,1024]` | bf16 |
| 2 | `02_ss_noise.npy` | Initial dense noise into sparse-structure flow | `[1,Cin,16,16,16]` | f32 |
| 3 | `03_ss_latent.npy` | z_s after FlowEuler diffusion (sparse-structure flow) | same as noise | f32 |
| 3 | `03_ss_pred_x_t.step{NNN}.npy` | (per-step, optional) intermediate diffusion samples | same as noise | f32 |
| 4 | `04_ss_decoder_logits.npy` | Pre-threshold logits from `sparse_structure_decoder` | `[1,1,R,R,R]` | f32 |
| 5 | `05_ss_coords.npy` | Occupancy coords `(B,x,y,z)` after `>0` threshold + max-pool to res 32 | `[N,4]` | int32 |
| 6 | `06_shape_slat_noise_feats.npy` | Initial noise feats for shape SLat (coords from step 5) | `[N,Cin]` | f32 |
| 7 | `07_shape_slat_raw_feats.npy` | Shape SLat feats after diffusion, **before** denormalization | `[N,Cout]` | f32 |
| 7 | `07_shape_slat_pred_x_t.step{NNN}.feats.npy` | (optional) per-step shape SLat feats | `[N,Cout]` | f32 |
| 8 | `08_shape_slat_denorm_feats.npy` | `raw * std + mean` (final shape SLat) | `[N,Cout]` | f32 |
| 9 | `09_tex_slat_noise_feats.npy` | Initial noise feats for tex SLat (coords reused from shape SLat) | `[N,Cin-Cshape]` | f32 |
| 10 | `10_tex_concat_cond_feats.npy` | Re-normalized shape feats fed as `concat_cond` into tex flow | `[N,Cshape]` | f32 |
| 11 | `11_tex_slat_raw_feats.npy` | Tex SLat feats after diffusion, **before** denormalization | `[N,Cout]` | f32 |
| 11 | `11_tex_slat_pred_x_t.step{NNN}.feats.npy` | (optional) per-step tex SLat feats | `[N,Cout]` | f32 |
| 12 | `12_tex_slat_denorm_feats.npy` | Final tex SLat feats | `[N,Cout]` | f32 |
| 13 | `13_mesh_vertices.npy` | Mesh vertices in world coords from shape decoder | `[V,3]` | f32 |
| 13 | `13_mesh_faces.npy` | Triangle indices | `[F,3]` | int |
| 14 | `14_shape_sub{i}.coords/.feats.npy` | Substructure SparseTensors returned with `return_subs=True` (used to guide tex decoder) | `[Ni,4]` / `[Ni,Ci]` | int / f32 |
| 15 | `15_tex_voxels.coords/.feats.npy` | Decoded PBR voxels: `feats[:, :3]=basecolor` (in [0,1]), `[3]=metallic`, `[4]=roughness`, `[5]=alpha` | `[N,4]` / `[N,6]` | int / f32 |

`manifest.json` records, for every `.npy`: shape, dtype, sha256, byte size,
and basic statistics (`min/max/mean/std`).

## Diffing the ROCm path against this reference

The ROCm path must reach each boundary with the same inputs. Bisect
top-down: a divergence at boundary *N* localizes the bug to the module
between *N-1* and *N*.

### Quick numeric check

```python
import numpy as np

def cmp(name, ref_path, rocm_path, atol=1e-4, rtol=1e-3):
    ref  = np.load(ref_path)
    test = np.load(rocm_path)
    assert ref.shape == test.shape, f'{name}: shape {ref.shape} vs {test.shape}'
    diff = np.abs(ref.astype(np.float32) - test.astype(np.float32))
    rel  = diff / (np.abs(ref).astype(np.float32) + 1e-9)
    print(f'{name:40s} max|d|={diff.max():.3e}  mean|d|={diff.mean():.3e}  '
          f'max_rel={rel.max():.3e}  ok={np.allclose(ref, test, atol=atol, rtol=rtol)}')

cmp('dinov3_cond',   'verify-dumps/01_dinov3_cond_512.npy', 'rocm-dumps/01_dinov3_cond_512.npy', atol=5e-3)
cmp('ss_latent',     'verify-dumps/03_ss_latent.npy',       'rocm-dumps/03_ss_latent.npy')
cmp('ss_logits',     'verify-dumps/04_ss_decoder_logits.npy','rocm-dumps/04_ss_decoder_logits.npy')
cmp('shape_raw',     'verify-dumps/07_shape_slat_raw_feats.npy', 'rocm-dumps/07_shape_slat_raw_feats.npy')
cmp('tex_raw',       'verify-dumps/11_tex_slat_raw_feats.npy',   'rocm-dumps/11_tex_slat_raw_feats.npy')
cmp('tex_voxels',    'verify-dumps/15_tex_voxels.feats.npy',     'rocm-dumps/15_tex_voxels.feats.npy')
```

### Tolerance bands (suggested starting points)

These reflect realistic platform spread (BF16 attention, FP16 GEMM, sparse
conv reorderings). Tighten once you have a clean baseline.

| Tensor | Suggested `atol` | Notes |
|---|---|---|
| `01_dinov3_cond_512` | `5e-3` | bf16 ViT — relax further if attention backends differ |
| `02_ss_noise`, `06_*_noise`, `09_*_noise` | exact | Same `torch.manual_seed(42)`; if these differ, RNG state ordering is wrong |
| `03_ss_latent`, `07_shape_slat_raw_feats`, `11_tex_slat_raw_feats` | `1e-3..1e-2` | Diffusion accumulates per-step error |
| `04_ss_decoder_logits` | `5e-3` | Threshold at 0; sign agreement matters more than magnitude |
| `05_ss_coords` | exact (sorted) | Compare as a *set* of tuples — argwhere ordering may differ |
| `13_mesh_vertices`, `13_mesh_faces` | exact (topology) | Vertex order may differ; compare bounding box + face count + by KD-tree |
| `15_tex_voxels.feats` | `1e-2` | Final PBR — large divergence here ⇒ broken textures |

### Sparse-tensor caveats

`coords` rows have no canonical ordering — sort lexicographically before
diffing, and reindex `feats` accordingly:

```python
def canon(coords, feats):
    key = np.lexsort(coords.T[::-1])
    return coords[key], feats[key]
```

For `.npy` SparseTensor pairs (`*.coords.npy` + `*.feats.npy`) where the row
*sets* might agree but ordering differs, this is the only meaningful diff.

### Step-by-step diffusion bisection

Run the dumper with `DUMP_PER_STEP=1` on both paths. The first step where
`pred_x_t.step{NNN}` diverges is the first DiT call that misbehaves —
typically that pinpoints a single attention or MLP kernel.

## Determinism notes

- `torch.manual_seed(seed)` is called *before* `get_cond` so DINOv3 dropout
  (none expected at eval, but be safe) is reproducible.
- The dumper re-seeds before each diffusion's noise sample (`seed`,
  `seed+1`, `seed+2`) to make individual stages reproducible even if you
  later run them in isolation. The original `pipeline.run()` does **not**
  re-seed, so `02_ss_noise` is identical between the two; later noise
  tensors differ unless the ROCm path also re-seeds in the same way.
- Sparse-conv kernels (FlexGEMM) and flash-attn are deterministic at the
  kernel level for fixed inputs. cuBLAS reductions can introduce small
  per-platform noise — this is what the tolerance bands above cover.

## File layout

```
verify-dumps/
├── README.md          (this file)
├── manifest.json      (generated; per-entry shape/dtype/sha256/stats)
├── 00_image_preprocessed.npy
├── 01_dinov3_cond_512.npy
├── 01_dinov3_neg_cond_512.npy
├── 02_ss_noise.npy
├── 03_ss_latent.npy
├── 04_ss_decoder_logits.npy
├── 05_ss_coords.npy
├── 06_shape_slat_noise_feats.npy
├── 07_shape_slat_raw_feats.npy
├── 08_shape_slat_denorm_feats.npy
├── 09_tex_slat_noise_feats.npy
├── 10_tex_concat_cond_feats.npy
├── 11_tex_slat_raw_feats.npy
├── 12_tex_slat_denorm_feats.npy
├── 13_mesh_vertices.npy
├── 13_mesh_faces.npy
├── 14_shape_sub{0..}.{coords,feats}.npy
└── 15_tex_voxels.{coords,feats}.npy
```

## Per-layer (per-block) dumps — `--dump-per-block`

Stage-boundary dumps localize a divergence to a *stage*; to localize it to a
single transformer block or decoder layer, regenerate with per-block capture:

```bash
# per-DiT-block + per-decoder-layer activations, shape mesh at 4 resolutions
DUMP_PER_BLOCK=1 DECODER_RES=64,128,256,512 ./run_dump_ground_truth.sh
```

This adds a `per_layer/` subdir (with its own `manifest.json`) and a few
single-step *input* tensors next to the stage dumps. The diffusion **samplers
are not touched** — instead each model is run **once** at a fixed `t=0.5` with
positive cond, and a forward hook captures every block's output. A HIP/CUDA
port reproduces the identical single forward from the dumped inputs and diffs
block-by-block. (`f32` on disk even though blocks compute in bf16 — capture is
the real compute, upcast losslessly for diffing.)

Single-step inputs (alongside the stage dumps):

| File | Model | What |
|---|---|---|
| `02b_ss_dit_step_{t,velocity}` | SS DiT | t=0.5; output velocity (inputs: `02_ss_noise` + `01_dinov3_cond_512`) |
| `06b_slat_dit_step_{x_t,coords,t,cond,velocity}` | shape-SLAT DiT | single-step input/output |
| `10b_tex_dit_step_{x_t,coords,t,velocity}` | tex-SLAT DiT | `x_t = cat(noise, shape_norm)` |

`per_layer/` entries (each `.feats` is `[N or tokens, C]`, `f32`):

| Prefix | Model | Per-entry |
|---|---|---|
| `ss_block{00..29}` | SS DiT (dense) | `[tokens, 1536]` block output |
| `shape_block{00..29}.feats` (+ `shape.coords`) | shape-SLAT DiT | sparse block output |
| `tex_block{00..29}.feats` (+ `tex.coords`) | tex-SLAT DiT | sparse block output |
| `shapedec_{from_latent,output_layer,L{i}_b{j}}.feats` (+ `shapedec_L{i}.coords`) | shape SC-VAE | per-layer; outer `L{i}` = resolution level 32→64→128→256→512 (matches `14_shape_sub*`); `*.subpred` = the per-level subdivision prediction |
| `texdec_{from_latent,output_layer,L{i}_b{j}}.feats` (+ `texdec_L{i}.coords`) | tex SC-VAE | per-layer |

Multi-res note: the SC-VAE network is resolution-independent — `set_resolution()`
only changes the `flexible_dual_grid_to_mesh` `grid_size`. So per-layer
activations are captured **once** (at the native res), and only the final mesh
is re-extracted per `DECODER_RES`, emitted as `13_mesh_r{res}_{vertices,faces}`
(plus an un-suffixed canonical `13_mesh_{vertices,faces}` = the 512 mesh).

## Comparison tool — `compare_dumps.py`

Diffs a reference dump dir against another (e.g. a HIP dump dir from another
GPU), manifest/glob-driven, with SparseTensor coord-canonicalization:

```bash
# stage boundaries
./compare_dumps.py --ref verify-dumps --test /path/to/rocm-dumps
# include the per_layer/ subdir (block-by-block bisection)
./compare_dumps.py --ref verify-dumps --test /path/to/rocm-dumps --per-layer
```

Reports `rel_L2 / max_abs / cosine / ok` per entry and prints the **first
divergent entry** (the layer to investigate). `*.coords` are compared as integer
row *sets*; `*.feats` are lexsorted by their coords before diffing (so a
different sparse row order is not flagged as a mismatch). Tune `--atol/--rtol`;
see the tolerance bands above.

> **TODO (rdna4 side):** mirror `--dump-per-block` / `--decoder-res` into
> `rdna4/trellis2/runner/dump_rocm.py` (it already has the `06b`/`10b`
> single-step captures and the same `dump_sparse` helpers) so the HIP path emits
> a matching `per_layer/` for `compare_dumps.py`.
