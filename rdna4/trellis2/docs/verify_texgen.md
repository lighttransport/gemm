# Texgen-only verification (CUDA → ROCm)

Goal: isolate the texture-baking + inpaint + GLB-export tail of
`o_voxel.postprocess.to_glb` (lines ~195–323) from the upstream
mesh-cleanup chain (fill_holes / simplify / non-manifold repair / UV
unwrap), so the **rdna4 cumesh_xatlas shim** can be validated against
CUDA without UV-unwrap nondeterminism muddying the comparison.

The cracks/wobble seen in the e2e ROCm GLB likely come from one of:
- our `CuMesh.simplify` (fast_simplification vs. cumesh's quadric path),
- our `CuMesh.repair_non_manifold_edges` (currently a no-op),
- our `CuMesh.unify_face_orientations` (trimesh.repair vs. cumesh),
- our `cuBVH.unsigned_distance` (trimesh proximity vs. cumesh BVH),
- our `texgen_sw_rast` (numba CPU rasterizer vs. nvdiffrast).

This procedure pins everything **upstream of the BVH+raster step** to
CUDA outputs and re-runs only the rdna4 baking tail. A diff against the
CUDA reference GLB tells us whether the cracks are in (a) baking/raster
or (b) mesh cleanup. Run both modes (full e2e and this isolated mode)
to localise the bug.

---

## CUDA-side dumps

Modify `cuda/trellis2/dump_ground_truth.py` (or its texgen equivalent)
to `np.save` the artefacts below into `cuda/trellis2/verify-dumps/`.
Patch the references to lines inside
`cpu/trellis2/trellis2_repo/o-voxel/o_voxel/postprocess.py` — the
variables already exist there, no recomputation needed.

### Required

Post-simplify, post-UV-unwrap mesh state (input to the baking tail):

| File                | Source in `to_glb`                  | Shape / dtype     |
|---------------------|-------------------------------------|-------------------|
| `uv_vertices.npy`   | `out_vertices` after L201           | `[V', 3] f32`     |
| `uv_faces.npy`      | `out_faces`     after L201          | `[F', 3] i32`     |
| `uv_uvs.npy`        | `out_uvs`       after L201          | `[V', 2] f32`     |
| `uv_vmaps.npy`      | `out_vmaps`     after L201          | `[V'] i64`        |
| `uv_normals.npy`    | `out_normals`   after L216          | `[V', 3] f32`     |

Pre-simplify mesh (the cuBVH source — used to map texels back to the
original surface for accurate attribute sampling):

| File                | Source in `to_glb`                       | Shape / dtype     |
|---------------------|------------------------------------------|-------------------|
| `bvh_vertices.npy`  | `vertices` (the original input arg)      | `[V, 3] f32`      |
| `bvh_faces.npy`     | `faces`    (the original input arg)      | `[F, 3] i32`      |

PBR attribute volume (the sparse voxels from `tex_voxels`):

| File                | Source in `to_glb`             | Shape / dtype       |
|---------------------|--------------------------------|---------------------|
| `attr_volume.npy`   | `attr_volume`                  | `[N, C] f32`        |
| `attr_coords.npy`   | `coords`                       | `[N, 3] i32`        |
| `attr_grid_size.npy`| `grid_size`                    | `[3] i32`           |

Constants (one JSON, easier than three loose npys):

```json
{
  "attr_layout": {
    "base_color": [0, 1, 2],
    "metallic":   3,
    "roughness":  4,
    "alpha":      5
  },
  "voxel_size":   1.953125e-3,
  "aabb":         [[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
  "texture_size": 1024,
  "remesh":       false
}
```

Save as `texgen_consts.json` alongside the npys. The exact `attr_layout`
must match `pipeline.pbr_attr_layout` from the CUDA run — print and
copy verbatim.

Reference GLB for binary/visual diff:

| File                  | Source                                      |
|-----------------------|---------------------------------------------|
| `textured_cuda.glb`   | the GLB produced by the CUDA `to_glb` run   |

### Optional (narrower bisects if cracks persist)

If the rdna4 baking path produces a different GLB even with all of the
above pinned, dump these to localise further:

| File                | Source in `to_glb`                 | Shape / dtype       |
|---------------------|------------------------------------|---------------------|
| `rast.npy`          | `rast` after the chunk loop L243   | `[1,H,W,4] f32`     |
| `valid_pos.npy`     | `valid_pos` at L250                | `[P, 3] f32`        |
| `bvh_face_id.npy`   | `face_id` from L254                | `[P] i64`           |
| `bvh_uvw.npy`       | `uvw`     from L254                | `[P, 3] f32`        |
| `attrs_pre_inpaint.npy` | `attrs` at L259 (before inpaint)| `[H, W, C] f32`     |
| `mask.npy`          | `mask` at L246                     | `[H, W] bool`       |

`rast.npy` lets us bypass `texgen_sw_rast` entirely (the numba CPU
rasterizer) and feed CUDA-rasterized texels into the BVH lookup.

---

## ROCm-side replay

Once the CUDA dumps are in `cuda/trellis2/verify-dumps/`, run:

```bash
cd /mnt/disk1/work/gemm/trellis2
rdna4/trellis2/.venv/bin/python rdna4/trellis2/tools/verify_texgen.py \
    --dump-dir cuda/trellis2/verify-dumps \
    --output-dir rdna4/trellis2/verify-dumps-rocm-texgen \
    [--use-cuda-rast]   # if rast.npy was dumped, skip texgen_sw_rast
```

The tool re-runs **only** the baking tail of `o_voxel.postprocess.to_glb`:

1. Load post-UV-unwrap mesh state (uv_*.npy).
2. Build `cumesh_xatlas.cuBVH` from `bvh_vertices/faces`.
3. Construct UV-clip-space positions (`uvs_rast` per L232).
4. Either rasterize via `texgen_sw_rast` (numba) **or** load `rast.npy`.
5. `dr.interpolate` → `valid_pos`.
6. `bvh.unsigned_distance(valid_pos, return_uvw=True)`.
7. `grid_sample_3d` for attrs at the corrected positions.
8. cv2 inpaint, PBR material, trimesh export → `textured_rocm.glb`.

Then diff:

```bash
rdna4/trellis2/.venv/bin/python rdna4/trellis2/tools/diff_glb.py \
    cuda/trellis2/verify-dumps/textured_cuda.glb \
    rdna4/trellis2/verify-dumps-rocm-texgen/textured_rocm.glb
```

The diff tool compares vertices/faces/uvs (must be identical — they
come from the same dump), and the baseColor/metallicRoughness textures
pixel-wise (max|d| and mean|d| per channel).

---

## What success looks like

- **Identical mesh tensors** (verts/faces/uvs from CUDA dump are byte-for-byte preserved).
- **PBR textures**: `max|d| ≤ 2/255` per channel, `mean|d| ≤ 0.5/255`.
  - Sub-pixel barycentric/raster differences can shift colour by ±1 LSB
    at chart boundaries; that's expected.
- **Visual**: load both GLBs in any viewer (`trimesh`, Blender,
  glTF-Sample-Viewer); textures should be visually indistinguishable.

If textures match but the **full e2e** ROCm run still has cracks, the
bug is in the upstream cumesh shim (simplify / non-manifold repair /
fix_normals / fill_holes), not in the baking tail.

---

## Failure modes to watch for

1. **`rast` differs heavily**: numba CPU rasterizer barycentrics differ
   from nvdiffrast at >1% of texels. Replace `texgen_sw_rast` (or pin
   to `--use-cuda-rast`).
2. **`bvh_face_id` differs at >1% of points**: trimesh proximity is
   approximate at silhouettes. Need a real BVH (e.g., `embree` via
   `trimesh.ray.ray_pyembree`, or port cumesh's BVH).
3. **`bvh_uvw` differs**: barycentric reconstruction in
   `cumesh_xatlas.cuBVH.unsigned_distance` is wrong. Check the
   `c1`↔`w2`, `c2`↔`w1` mapping (already verified to ~3e-8 against
   `trimesh.proximity.closest_point` on a unit sphere — but a stress
   test on the real mesh is warranted).
4. **`attrs` differs but `bvh_uvw` matches**: bug is in
   `o_voxel.utils.grid_sample_3d` on ROCm. Run a smaller repro with
   identical inputs.

---

## Why isolate

The ROCm e2e run produces:

- 1.5M-vert decoder mesh (matches CUDA: `cuda/trellis2/verify-dumps/13_mesh_vertices.npy` → `(1468404, 3)`),
- 80k-vert post-cleanup mesh (CUDA, with the real `cumesh.simplify` +
  manifold repair, lands at a different vertex count and topology),
- different UV charts (xatlas is run on different inputs),
- different baked textures.

Without pinning the post-UV-unwrap state, every difference is
ambiguous. Pinning isolates the **baking + BVH** code we wrote, which
is the smaller surface and the easier port to validate.
