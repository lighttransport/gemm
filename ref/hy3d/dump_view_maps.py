"""Dump 6-view normal + position maps from MeshRender for CUDA-side validation.

Mirrors what hy3dpaint's ViewProcessor.render_normal_multiview /
render_position_multiview produce as conditioning input for the texgen
multiview UNet. Use the .npy files as ground truth when porting the
view-map renderer to native CUDA (cuda/hy3d_paint/test_view_maps.c).

Default 6 views (front / right / back / left / top / bottom):
    azim = [  0,  90, 180, 270,   0, 180]
    elev = [  0,   0,   0,   0,  90, -90]

Output:
    {prefix}_view{V}_normal.npy    [H, W, 3] float32, ((n+1)*0.5) world-space
    {prefix}_view{V}_position.npy  [H, W, 3] float32, (0.5 - p/scale_factor)

Usage:
  uv run python dump_view_maps.py \
      --mesh /tmp/hy3d_final.obj \
      --hy3d-repo /mnt/disk01/models/Hunyuan3D-2.1-repo/hy3dshape \
      --outdir /tmp/hy3d_view_ref \
      [--resolution 512]
"""
import argparse
import os
import sys

import numpy as np
import trimesh


def _setup_env(hy3d_repo: str):
    if not os.path.isdir(hy3d_repo):
        sys.exit(f"ERROR: HY3D_REPO not a directory: {hy3d_repo}")
    hy3dpaint_dir = os.path.join(hy3d_repo, "..", "hy3dpaint")
    if not os.path.isdir(hy3dpaint_dir):
        sys.exit(f"ERROR: hy3dpaint dir not found at {hy3dpaint_dir}")
    sys.path.insert(0, hy3dpaint_dir)
    os.chdir(hy3dpaint_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="input mesh (.obj/.glb/.ply)")
    ap.add_argument("--hy3d-repo", default="/mnt/disk01/models/Hunyuan3D-2.1-repo/hy3dshape",
                    help="path to hy3dshape dir (sibling hy3dpaint must exist)")
    ap.add_argument("--outdir", default="/tmp/hy3d_view_ref")
    ap.add_argument("--resolution", type=int, default=512)
    ap.add_argument("--prefix", default="ref")
    args = ap.parse_args()

    _setup_env(args.hy3d_repo)
    from DifferentiableRenderer.MeshRender import MeshRender

    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    vtx = np.asarray(mesh.vertices, dtype=np.float32)
    tri = np.asarray(mesh.faces, dtype=np.int32)
    print(f"loaded {args.mesh}: {vtx.shape[0]} verts, {tri.shape[0]} tris")

    render = MeshRender(
        camera_distance=1.45,
        default_resolution=args.resolution,
        texture_size=args.resolution,
        bake_mode="back_sample",
        raster_mode="cr",
        shader_type="face",
    )
    render.set_mesh(vtx, tri, scale_factor=1.15, auto_center=True)

    azims = [0.0, 90.0, 180.0, 270.0, 0.0, 180.0]
    elevs = [0.0, 0.0, 0.0, 0.0, 90.0, -90.0]

    os.makedirs(args.outdir, exist_ok=True)
    for v, (elev, azim) in enumerate(zip(elevs, azims)):
        nrm = render.render_normal(elev, azim, use_abs_coor=True, return_type="np")
        pos = render.render_position(elev, azim, return_type="np")
        if hasattr(nrm, "cpu"):
            nrm = nrm.cpu().numpy()
        if hasattr(pos, "cpu"):
            pos = pos.cpu().numpy()
        nrm = np.asarray(nrm, dtype=np.float32)
        pos = np.asarray(pos, dtype=np.float32)
        n_path = os.path.join(args.outdir, f"{args.prefix}_view{v}_normal.npy")
        p_path = os.path.join(args.outdir, f"{args.prefix}_view{v}_position.npy")
        np.save(n_path, nrm)
        np.save(p_path, pos)
        cov = float((np.abs(pos - 1.0).sum(axis=-1) > 1e-3).mean())
        print(f"view {v}  elev={elev:+.0f} azim={azim:+.0f}  "
              f"shape={nrm.shape}  coverage={cov*100:.1f}%  "
              f"normal range=[{nrm.min():.3f}, {nrm.max():.3f}]  "
              f"pos range=[{pos.min():.3f}, {pos.max():.3f}]")
    print(f"wrote 12 .npy files to {args.outdir}")


if __name__ == "__main__":
    main()
