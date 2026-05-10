"""Replay only the texture-baking tail of o_voxel.postprocess.to_glb against
CUDA-pinned inputs and diff each stage. See docs/verify_texgen.md.

Usage:
    rdna4/trellis2/.venv/bin/python rdna4/trellis2/tools/verify_texgen.py \
        --dump-dir cuda/trellis2/verify-dumps \
        --output-dir rdna4/trellis2/verify-dumps-rocm-texgen \
        [--use-cuda-rast]

Stages diffed (each one localises a different shim component):
    rast    -> texgen_sw_rast (numba CPU rasterizer)
    bvh     -> cumesh_xatlas.cuBVH.unsigned_distance
    attrs   -> grid_sample_3d on ROCm
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Bring rdna4/trellis2/{shims,kernels,...} onto sys.path.
_HERE = Path(__file__).resolve()
_RDNA4 = _HERE.parents[1]
sys.path.insert(0, str(_RDNA4))
sys.path.insert(0, str(_RDNA4.parents[1] / 'cpu' / 'trellis2' / 'trellis2_repo'))


def _device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _load(dump_dir: Path, name: str) -> np.ndarray:
    return np.load(dump_dir / f'{name}.npy')


def _t(arr: np.ndarray, device: str = None, dtype=None) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if dtype is not None:
        t = t.to(dtype)
    if device is None:
        device = _device()
    return t.to(device)


def _stat(name: str, ours: torch.Tensor, ref: torch.Tensor, atol: float = 1e-5):
    o = ours.detach().cpu().float()
    r = ref.detach().cpu().float()
    if o.shape != r.shape:
        print(f'  [{name}] SHAPE MISMATCH ours={tuple(o.shape)} ref={tuple(r.shape)}')
        return
    d = (o - r).abs()
    eq = torch.equal(o, r)
    print(f'  [{name}] shape={tuple(o.shape)} '
          f'max|d|={d.max().item():.3e} mean|d|={d.mean().item():.3e} '
          f'exact={eq}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump-dir', required=True, help='dir with CUDA *.npy + textured_cuda.glb')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--use-cuda-rast', action='store_true',
                    help='Load rast.npy instead of running texgen_sw_rast.')
    ap.add_argument('--rast-chunk', type=int, default=100000)
    args = ap.parse_args()

    dump = Path(args.dump_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- shims ---
    from shims import cumesh_xatlas, texgen_sw_rast
    cumesh_xatlas.install_as_cumesh()
    texgen_sw_rast.install_as_nvdiffrast()
    import cumesh
    import nvdiffrast.torch as dr

    dev = _device()
    print(f'[device] {dev}')

    # --- constants ---
    with open(dump / 'texgen_consts.json') as f:
        consts = json.load(f)
    attr_layout = consts['attr_layout']
    voxel_size = float(consts['voxel_size'])
    aabb_l = consts['aabb']
    aabb = torch.tensor(aabb_l, device=dev, dtype=torch.float32)
    H = W = int(consts['texture_size'])
    print(f'[const] tex={H}x{W} voxel_size={voxel_size} aabb={aabb_l}')

    # --- load post-UV-unwrap mesh state (input to baking tail) ---
    out_vertices = _t(_load(dump, 'uv_vertices'),  dev, torch.float32)
    out_faces    = _t(_load(dump, 'uv_faces'),     dev, torch.int32)
    out_uvs      = _t(_load(dump, 'uv_uvs'),       dev, torch.float32)
    out_normals  = _t(_load(dump, 'uv_normals'),   dev, torch.float32)
    out_vmaps    = _t(_load(dump, 'uv_vmaps'),     dev, torch.int64)
    print(f'[uv]  V={out_vertices.shape[0]} F={out_faces.shape[0]}')

    # --- load BVH source (pre-simplify mesh) ---
    bvh_v_np = _load(dump, 'bvh_vertices')
    bvh_f_np = _load(dump, 'bvh_faces')
    bvh_v = _t(bvh_v_np, dev, torch.float32)
    bvh_f = _t(bvh_f_np, dev, torch.int64)
    print(f'[bvh] V={bvh_v.shape[0]} F={bvh_f.shape[0]}')

    # --- load PBR voxels ---
    attr_volume_np = _load(dump, 'attr_volume')
    attr_coords_np = _load(dump, 'attr_coords')
    attr_volume = _t(attr_volume_np, dev, torch.float32)
    attr_coords = _t(attr_coords_np, dev, torch.int32)
    grid_size_np = _load(dump, 'attr_grid_size').astype(np.int64)
    print(f'[attr] N={attr_volume.shape[0]} C={attr_volume.shape[1]} grid={grid_size_np.tolist()}')

    # =======================================================================
    # Stage 1: rasterize UVs -> rast (texel -> face_id, bary)
    # =======================================================================
    rast_ref = _t(_load(dump, 'rast'), dev, torch.float32)
    if args.use_cuda_rast:
        print('[rast] --use-cuda-rast: using rast.npy verbatim')
        rast = rast_ref.clone()
    else:
        print('[rast] running texgen_sw_rast (numba CPU)')
        ctx = dr.RasterizeCudaContext()
        uvs_rast = torch.cat([
            out_uvs * 2 - 1,
            torch.zeros_like(out_uvs[:, :1]),
            torch.ones_like(out_uvs[:, :1]),
        ], dim=-1).unsqueeze(0)
        rast = torch.zeros((1, H, W, 4), device=dev, dtype=torch.float32)
        t0 = time.time()
        for i in range(0, out_faces.shape[0], args.rast_chunk):
            chunk_faces = out_faces[i:i + args.rast_chunk]
            chunk_rast, _ = dr.rasterize(ctx, uvs_rast, chunk_faces, resolution=[H, W])
            m_chunk = chunk_rast[..., 3:4] > 0
            chunk_rast[..., 3:4] = chunk_rast[..., 3:4] + i
            rast = torch.where(m_chunk, chunk_rast, rast)
        print(f'[rast] done in {time.time()-t0:.1f}s')
        _stat('rast', rast, rast_ref)
    np.save(out / 'rast_rocm.npy', rast.cpu().numpy())

    mask = rast[0, ..., 3] > 0
    mask_ref = _t(_load(dump, 'mask'), dev, torch.bool)
    _stat('mask (as float)', mask.float(), mask_ref.float())

    # =======================================================================
    # Stage 2: interpolate -> 3D pos per texel
    # =======================================================================
    pos = dr.interpolate(out_vertices.unsqueeze(0), rast, out_faces)[0][0]
    valid_pos = pos[mask]
    valid_pos_ref = _t(_load(dump, 'valid_pos'), dev, torch.float32)
    _stat('valid_pos', valid_pos, valid_pos_ref)
    np.save(out / 'valid_pos_rocm.npy', valid_pos.cpu().numpy())

    # =======================================================================
    # Stage 3: BVH lookup on the original (pre-simplify) mesh
    # =======================================================================
    bvh = cumesh.cuBVH(bvh_v, bvh_f)
    t0 = time.time()
    _, face_id, uvw = bvh.unsigned_distance(valid_pos, return_uvw=True)
    print(f'[bvh] unsigned_distance done in {time.time()-t0:.1f}s')
    face_id_ref = _t(_load(dump, 'bvh_face_id'), dev, torch.int64)
    uvw_ref     = _t(_load(dump, 'bvh_uvw'),     dev, torch.float32)
    _stat('bvh.face_id (as float)', face_id.float(), face_id_ref.float())
    _stat('bvh.uvw',                uvw,            uvw_ref)
    np.save(out / 'bvh_face_id_rocm.npy', face_id.cpu().numpy())
    np.save(out / 'bvh_uvw_rocm.npy',     uvw.cpu().numpy())

    # =======================================================================
    # Stage 4: pull texel -> orig surface 3D pos -> attr sample
    # =======================================================================
    bvh_f_long = bvh_f if bvh_f.dtype == torch.int64 else bvh_f.long()
    orig_tri_verts = bvh_v[bvh_f_long[face_id.long()]]  # (P, 3, 3)
    valid_pos_corr = (orig_tri_verts * uvw.unsqueeze(-1)).sum(dim=1)

    from flex_gemm.ops.grid_sample import grid_sample_3d
    C = attr_volume.shape[1]
    attrs = torch.zeros(H, W, C, device=dev)
    grid = ((valid_pos_corr - aabb[0]) / voxel_size).reshape(1, -1, 3)
    coords_pad = torch.cat([
        torch.zeros_like(attr_coords[:, :1]), attr_coords
    ], dim=-1)
    attrs[mask] = grid_sample_3d(
        attr_volume,
        coords_pad,
        shape=torch.Size([1, C, *grid_size_np.tolist()]),
        grid=grid,
        mode='trilinear',
    )
    attrs_ref = _t(_load(dump, 'attrs_pre_inpaint'), dev, torch.float32)
    _stat('attrs_pre_inpaint', attrs, attrs_ref)
    np.save(out / 'attrs_pre_inpaint_rocm.npy', attrs.cpu().numpy())

    # =======================================================================
    # Stage 5: inpaint + PBR material + GLB export
    # =======================================================================
    import cv2
    import trimesh
    import trimesh.visual
    from PIL import Image

    mask_np = mask.cpu().numpy()
    base_color = np.clip(attrs[..., attr_layout['base_color']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    metallic   = np.clip(attrs[..., attr_layout['metallic']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    roughness  = np.clip(attrs[..., attr_layout['roughness']].cpu().numpy() * 255, 0, 255).astype(np.uint8)
    alpha      = np.clip(attrs[..., attr_layout['alpha']].cpu().numpy() * 255, 0, 255).astype(np.uint8)

    mask_inv = (~mask_np).astype(np.uint8)
    base_color = cv2.inpaint(base_color, mask_inv, 3, cv2.INPAINT_TELEA)
    metallic   = cv2.inpaint(metallic,   mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    roughness  = cv2.inpaint(roughness,  mask_inv, 1, cv2.INPAINT_TELEA)[..., None]
    alpha      = cv2.inpaint(alpha,      mask_inv, 1, cv2.INPAINT_TELEA)[..., None]

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(np.concatenate([base_color, alpha], axis=-1)),
        baseColorFactor=np.array([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(np.concatenate([np.zeros_like(metallic), roughness, metallic], axis=-1)),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode='OPAQUE',
        doubleSided=False if consts.get('remesh', False) else True,
    )

    v_np = out_vertices.cpu().numpy().copy()
    f_np = out_faces.cpu().numpy()
    uv_np = out_uvs.cpu().numpy().copy()
    n_np = out_normals.cpu().numpy().copy()
    v_np[:, 1], v_np[:, 2] = v_np[:, 2].copy(), -v_np[:, 1].copy()
    n_np[:, 1], n_np[:, 2] = n_np[:, 2].copy(), -n_np[:, 1].copy()
    uv_np[:, 1] = 1 - uv_np[:, 1]

    glb = trimesh.Trimesh(
        vertices=v_np, faces=f_np, vertex_normals=n_np, process=False,
        visual=trimesh.visual.TextureVisuals(uv=uv_np, material=material),
    )
    out_glb = out / 'textured_rocm.glb'
    glb.export(str(out_glb))
    print(f'[glb] wrote {out_glb}')


if __name__ == '__main__':
    main()
