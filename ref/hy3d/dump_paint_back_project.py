"""Dump a single-view back-projection oracle for cuda/hy3d_paint validation.

This is a self-contained numpy port of MeshRender.back_project (back_sample
mode) plus the texture-space prerequisites (extract_textiles equivalent).
We avoid the upstream custom_rasterizer C++ extension entirely so this runs
in any plain torch/numpy env.

Pipeline mirrored:
  1. Build mesh (trimesh icosphere).
  2. UV unwrap with xatlas.
  3. Apply MeshRender.set_mesh coord transform: negate XY, swap YZ,
     v=1-v, auto-center+scale to scale_factor=1.15.
  4. extract_textiles equivalent: rasterize in UV space → tex_pos[Htex,Wtex,3]
     + tex_cov[Htex,Wtex] (dense layout, unlike MeshRender's sparse
     [N_visible,4] tex_position; the C kernel back_project_sample_f32
     consumes dense).
  5. Render view (elev,azim,camera_distance,ortho_scale) → visible_mask,
     depth (camera-space z, NOT normalized), normal (per-pixel face normal),
     cos_image = -normal.z (lookat = (0,0,-1)), sketch=0 (skip the
     boundary-shrink path).
  6. back_project_sample (numpy port): write ref_texture/ref_cos_map.

Outputs (all f32 .npy unless noted):
  mesh_meta.json                        — Htex, Wtex, Himg, Wimg, C, depth_thres
  tex_pos.npy   [Htex,Wtex,3]
  tex_cov.npy   [Htex,Wtex] int32
  image.npy     [Himg,Wimg,C]
  depth.npy     [Himg,Wimg]
  visible.npy   [Himg,Wimg]
  cos.npy       [Himg,Wimg]
  w2c.npy       [16] column-major
  proj.npy      [2]   (proj00, proj11)
  ref_tex.npy   [Htex,Wtex,C]
  ref_cos.npy   [Htex,Wtex]

Run:
  uv run --with trimesh --with xatlas --with numpy \\
      python ref/hy3d/dump_paint_back_project.py \\
      --outdir /tmp/hy3d_paint_bp_ref
"""
import argparse
import json
import os
import sys

import numpy as np
import trimesh
import xatlas


def get_mv_matrix(elev_deg, azim_deg, dist, center=(0.0, 0.0, 0.0)):
    """Match MeshRender.camera_utils.get_mv_matrix (lookat from elev/azim).

    Camera is placed at (cx + d*cos(e)*sin(a), cy + d*sin(e), cz + d*cos(e)*cos(a))
    looking at center, +Y up. Returns 4x4 row-major world->camera.
    Done in float64 (the elev=±90 ties tipped views in cuda/hy3d_paint —
    same gotcha as test_view_maps).
    """
    e = np.deg2rad(np.float64(elev_deg))
    a = np.deg2rad(np.float64(azim_deg))
    d = np.float64(dist)
    cx, cy, cz = (np.float64(v) for v in center)
    eye = np.array([cx + d * np.cos(e) * np.sin(a),
                    cy + d * np.sin(e),
                    cz + d * np.cos(e) * np.cos(a)], dtype=np.float64)
    target = np.array([cx, cy, cz], dtype=np.float64)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    up0 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(fwd, up0); right /= np.linalg.norm(right)
    up = np.cross(right, fwd)
    R = np.stack([right, up, -fwd], 0)  # world->cam rotation
    t = -R @ eye
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = t
    return M.astype(np.float32)


def get_ortho_proj(scale):
    """Diagonal proj for ortho box [-scale, scale]^3 → NDC [-1,1]^3.
    Matches MeshRender.get_orthographic_projection_matrix's diagonal.
    Returns the full 4x4; back_project_sample only uses proj[0,0], proj[1,1]."""
    s = np.float32(1.0 / scale)
    P = np.eye(4, dtype=np.float32)
    P[0, 0] = s
    P[1, 1] = s
    return P


def normalize_mesh(vtx_pos, scale_factor=1.15):
    """Mirror MeshRender.set_mesh: negate XY, swap YZ, auto-center+scale."""
    v = vtx_pos.astype(np.float32).copy()
    v[:, [0, 1]] = -v[:, [0, 1]]
    v[:, [1, 2]] = v[:, [2, 1]]
    mx = v.max(0); mn = v.min(0)
    c = (mx + mn) * 0.5
    scale = np.linalg.norm(v - c, axis=1).max() * 2.0
    v = (v - c) * (scale_factor / float(scale))
    return v


def rasterize_attr(verts_2d_or_clip, faces, attrs, H, W,
                   per_face_attr=False, ndc_to_pixel=True):
    """Rasterize triangles into an HxW image, interpolating attrs per pixel.

    verts_2d_or_clip: [N,2] in [-1,1] (we treat z=0).
    faces: [F,3] int.
    attrs: [N, K] (per-vertex) OR [F, K] (per-face if per_face_attr).
    Returns:
        out_attr [H,W,K] f32 (zeros where uncovered)
        cov      [H,W]   int32 (1 if covered)
        face_id  [H,W]   int32 (-1 if uncovered)

    Pure numpy. Edge-function rasterizer with top-left fill rule. Slow O(F*H*W)
    in the worst case; fine for icosphere (320 faces) at 256² (~21M ops).
    """
    if ndc_to_pixel:
        # NDC [-1,1] → pixel center coords. NO Y flip: matches the kernel's
        # back_project_sample_f32 sampling formula `fy = (py*0.5+0.5)*Himg`,
        # i.e. image row 0 corresponds to NDC y=-1.
        px = (verts_2d_or_clip[:, 0] * 0.5 + 0.5) * W - 0.5
        py = (verts_2d_or_clip[:, 1] * 0.5 + 0.5) * H - 0.5
    else:
        px = verts_2d_or_clip[:, 0]
        py = verts_2d_or_clip[:, 1]
    pts = np.stack([px, py], -1)  # [N,2]

    K = attrs.shape[-1]
    out = np.zeros((H, W, K), dtype=np.float32)
    cov = np.zeros((H, W), dtype=np.int32)
    fid = np.full((H, W), -1, dtype=np.int32)
    depth = np.full((H, W), np.inf, dtype=np.float32)  # for z-buffer if attrs include z

    xx, yy = np.meshgrid(np.arange(W), np.arange(H))
    xx = xx.astype(np.float32); yy = yy.astype(np.float32)

    for f in range(faces.shape[0]):
        i0, i1, i2 = faces[f]
        x0, y0 = pts[i0]; x1, y1 = pts[i1]; x2, y2 = pts[i2]
        # Bounding box (clamped)
        xmin = max(int(np.floor(min(x0, x1, x2))), 0)
        xmax = min(int(np.ceil (max(x0, x1, x2))), W - 1)
        ymin = max(int(np.floor(min(y0, y1, y2))), 0)
        ymax = min(int(np.ceil (max(y0, y1, y2))), H - 1)
        if xmax < xmin or ymax < ymin:
            continue
        # Edge function denom (signed area * 2)
        denom = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(denom) < 1e-12:
            continue
        # Vectorize across the tile
        xt = xx[ymin:ymax+1, xmin:xmax+1]
        yt = yy[ymin:ymax+1, xmin:xmax+1]
        w0 = ((x1 - xt) * (y2 - yt) - (x2 - xt) * (y1 - yt)) / denom
        w1 = ((x2 - xt) * (y0 - yt) - (x0 - xt) * (y2 - yt)) / denom
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        if per_face_attr:
            interp = np.broadcast_to(attrs[f][None, None, :], w0.shape + (K,)).copy()
        else:
            a0 = attrs[i0]; a1 = attrs[i1]; a2 = attrs[i2]
            interp = (w0[..., None] * a0 + w1[..., None] * a1 + w2[..., None] * a2)
        # Z-buffer if K==1 won't apply; instead always replace if face closer.
        # We pass attrs as concatenated [interp_attrs, z] so caller can z-test
        # outside. For now: simple "first writer wins" if cov=0, else compare
        # the first attribute as a tiebreaker is wrong. So expect caller to
        # pass z as the LAST attr column and we do z-test on that.
        z = interp[..., -1]
        write = inside & (z < depth[ymin:ymax+1, xmin:xmax+1])
        if not write.any():
            continue
        depth[ymin:ymax+1, xmin:xmax+1] = np.where(
            write, z, depth[ymin:ymax+1, xmin:xmax+1])
        out[ymin:ymax+1, xmin:xmax+1] = np.where(
            write[..., None], interp, out[ymin:ymax+1, xmin:xmax+1])
        cov[ymin:ymax+1, xmin:xmax+1] = np.where(
            write, 1, cov[ymin:ymax+1, xmin:xmax+1])
        fid[ymin:ymax+1, xmin:xmax+1] = np.where(
            write, f, fid[ymin:ymax+1, xmin:xmax+1])
    return out, cov, fid


def extract_tex_position_dense(vtx_pos, vtx_uv, faces_pos, faces_uv, Htex, Wtex):
    """UV-space rasterize. Returns dense tex_pos[Htex,Wtex,3] and tex_cov[Htex,Wtex].

    Matches MeshRender.extract_textiles' rast_out + interpolate(self.vtx_pos),
    but stored densely (the C kernel back_project_sample_f32 indexes dense).
    UV is in [0,1] - shift to NDC [-1,1] (matches MeshRender's *2-1).
    """
    uv_ndc = vtx_uv * 2.0 - 1.0  # [N_uv, 2]
    # Attribute = vertex world position (xyz). We need per-uv-vertex pos. The
    # mesh has separate UV verts (xatlas's uv_idx) but pos_idx-aligned vert
    # array via xatlas's vmap. Build per-uv-vert pos with vmap.
    # vmap[i] = original vertex index for uv-vert i.
    # Caller provides vtx_pos already gathered: vtx_pos[vmap].
    # Stack a z=1 dummy column to use as depth tiebreaker (UV is flat so
    # all faces share z=1; first writer wins, fine for non-overlapping atlas).
    attrs = np.concatenate([vtx_pos, np.ones((vtx_pos.shape[0], 1), dtype=np.float32)], -1)
    # In UV rasterize, "faces" use uv_idx (which already indexes into vtx_pos
    # since we passed vtx_pos pre-gathered with vmap)
    out, cov, fid = rasterize_attr(uv_ndc, faces_uv, attrs, Htex, Wtex)
    tex_pos = out[..., :3].copy()
    return tex_pos, cov


def render_view(vtx_pos, faces, w2c, proj, H, W):
    """Render visible_mask, depth(cam-z), per-pixel face normal."""
    # Transform to camera + clip space
    N = vtx_pos.shape[0]
    homo = np.concatenate([vtx_pos, np.ones((N, 1), dtype=np.float32)], -1)
    cam = (w2c @ homo.T).T  # [N,4]
    clip = (proj @ cam.T).T  # [N,4]
    ndc = clip[:, :3] / clip[:, 3:4]
    # face normals (camera-space)
    p0 = cam[faces[:, 0], :3]
    p1 = cam[faces[:, 1], :3]
    p2 = cam[faces[:, 2], :3]
    fn = np.cross(p1 - p0, p2 - p0)
    fn /= np.maximum(np.linalg.norm(fn, axis=-1, keepdims=True), 1e-20)
    # Rasterize: attrs = [face_normal_xyz, cam_z]. Use per-face attr for normal,
    # per-vertex z for depth. Combine into one shot: rasterize per-face normal
    # AND per-vertex z separately, then merge.
    # Interpolated z (per-vertex): pass cam_z_per_vert as last col.
    z_attr = cam[:, 2:3]  # [N,1]
    # We need both: per-face normal + per-vertex z. Run rasterize twice on the
    # same geometry — first pass for z + face_id (use z as depth tiebreaker),
    # second pass uses fid map to fill face normal.
    z_out, cov, fid = rasterize_attr(ndc[:, :2], faces, z_attr, H, W)
    depth_img = z_out[..., 0]  # cam-z
    # Per-pixel face normal from fid
    normal_img = np.zeros((H, W, 3), dtype=np.float32)
    mask = fid >= 0
    normal_img[mask] = fn[fid[mask]]
    visible = cov.astype(np.float32)
    return visible, depth_img, normal_img


def back_project_numpy(tex_pos, tex_cov, image, depth, visible, cos_img,
                       w2c, proj00, proj11, depth_thres=3e-3):
    """Numpy port of cuda kernel back_project_sample_f32 + MeshRender's
    matching back_sample path. Bilinear RGB sampling, depth + cos test."""
    Htex, Wtex, _ = tex_pos.shape
    Himg, Wimg, C = image.shape
    out_tex = np.zeros((Htex, Wtex, C), dtype=np.float32)
    out_cos = np.zeros((Htex, Wtex), dtype=np.float32)

    # Flatten texels
    p = tex_pos.reshape(-1, 3)  # [Nt,3]
    cov = tex_cov.reshape(-1)
    Nt = p.shape[0]
    # World -> camera (w2c is row-major 4x4; apply to row vec)
    homo = np.concatenate([p, np.ones((Nt, 1), dtype=np.float32)], -1)
    cam = (w2c @ homo.T).T
    cx = cam[:, 0]; cy = cam[:, 1]; cz = cam[:, 2]
    px = cx * proj00; py = cy * proj11
    pz = cz
    # Reject outside [-1,1]^2 OR uncovered
    valid = (cov > 0) & (px >= -1) & (px <= 1) & (py >= -1) & (py <= 1)
    fx = (px * 0.5 + 0.5) * Wimg
    fy = (py * 0.5 + 0.5) * Himg
    ix = np.clip(fx.astype(np.int32), 0, Wimg - 1)
    iy = np.clip(fy.astype(np.int32), 0, Himg - 1)
    wx = fx - ix
    wy = fy - iy
    ix1 = np.clip(ix + 1, 0, Wimg - 1)
    iy1 = np.clip(iy + 1, 0, Himg - 1)
    i00 = iy * Wimg + ix
    i01 = iy * Wimg + ix1
    i10 = iy1 * Wimg + ix
    i11 = iy1 * Wimg + ix1
    # Match kernel's iy convention: kernel stores image as row-major Himg×Wimg
    # with the same Y direction we're building here (image[0,0] is top-left,
    # py>0 → fy near top of image). render_view uses ndc Y flip (rasterize_attr
    # flips Y). Here we DO NOT flip because depth/visible/cos/image are all in
    # the same image-space coords.
    # Defer per-texel masking via boolean indices.
    d = depth.reshape(-1)[i00]
    vis = visible.reshape(-1)[i00]
    cw = cos_img.reshape(-1)[i00]
    valid &= vis > 0
    valid &= np.abs(pz - d) < depth_thres
    valid &= cw > 0
    img = image.reshape(-1, C)
    s00 = img[i00]; s01 = img[i01]; s10 = img[i10]; s11 = img[i11]
    samp = ((s00 * (1 - wx[:, None]) + s01 * wx[:, None]) * (1 - wy[:, None])
            + (s10 * (1 - wx[:, None]) + s11 * wx[:, None]) * wy[:, None])
    out_flat = out_tex.reshape(-1, C)
    cos_flat = out_cos.reshape(-1)
    out_flat[valid] = samp[valid]
    cos_flat[valid] = cw[valid]
    return out_tex, out_cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/hy3d_paint_bp_ref")
    ap.add_argument("--htex", type=int, default=128)
    ap.add_argument("--himg", type=int, default=128)
    ap.add_argument("--subdiv", type=int, default=2,
                    help="icosphere subdivision (320 faces at 2)")
    ap.add_argument("--elev", type=float, default=0.0)
    ap.add_argument("--azim", type=float, default=0.0)
    ap.add_argument("--dist", type=float, default=1.45)
    ap.add_argument("--ortho", type=float, default=1.2,
                    help="ortho half-extent (matches paint pipeline default)")
    ap.add_argument("--multiview", action="store_true",
                    help="Dump 4 views (azims 0/90/180/270 at elev 0) and the "
                         "weighted bake-blend reference. Adds bake_tex.npy / "
                         "bake_trust.npy + per-view view_<i>_*.npy.")
    ap.add_argument("--bake-exp", type=float, default=6.0,
                    help="cos exponent for bake blending (matches MeshRender)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print(f"icosphere subdivisions={args.subdiv}", file=sys.stderr)
    m = trimesh.creation.icosphere(subdivisions=args.subdiv)
    vtx_pos = m.vertices.astype(np.float32)
    faces = m.faces.astype(np.int32)
    print(f"  verts={vtx_pos.shape[0]} faces={faces.shape[0]}", file=sys.stderr)

    print("xatlas UV unwrap...", file=sys.stderr)
    vmap, uv_idx, uvs = xatlas.parametrize(vtx_pos, faces)
    # vmap maps uv-vert idx -> original vert idx; gather pos to uv-vert layout.
    vtx_pos_uv = vtx_pos[vmap]
    uv_idx = uv_idx.astype(np.int32)
    uvs = uvs.astype(np.float32)
    print(f"  uv verts={vtx_pos_uv.shape[0]} uv faces={uv_idx.shape[0]}", file=sys.stderr)

    # MeshRender coord transform — normalize ONCE on the full vtx_pos so the
    # auto-center/scale picks one consistent center, then gather into the UV
    # layout via vmap (otherwise the two normalize_mesh calls compute
    # different centers from different point sets and the texel positions
    # drift away from the rendered geometry, breaking the depth test).
    print("normalize mesh (negate XY, swap YZ, auto-center)", file=sys.stderr)
    vtx_pos_n = normalize_mesh(vtx_pos, scale_factor=1.15)
    vtx_pos_uv_n = vtx_pos_n[vmap]
    # Flip v (y) — match MeshRender.set_mesh's `vtx_uv[:,1] = 1 - vtx_uv[:,1]`.
    uvs_flip = uvs.copy()
    uvs_flip[:, 1] = 1.0 - uvs_flip[:, 1]

    print(f"extract_textiles dense rasterize {args.htex}x{args.htex}", file=sys.stderr)
    tex_pos, tex_cov = extract_tex_position_dense(
        vtx_pos_uv_n, uvs_flip, faces, uv_idx, args.htex, args.htex)
    print(f"  covered texels: {int(tex_cov.sum())} / {args.htex*args.htex}",
          file=sys.stderr)

    proj = get_ortho_proj(args.ortho)
    proj00 = float(proj[0, 0]); proj11 = float(proj[1, 1])
    cos_thres = float(np.cos(np.deg2rad(75.0)))

    def render_and_back_project(elev, azim, image_seed):
        w2c = get_mv_matrix(elev, azim, args.dist)
        visible, depth, normal = render_view(
            vtx_pos_n, faces, w2c, proj, args.himg, args.himg)
        cos_img = normal[..., 2].copy()
        cos_img[cos_img < cos_thres] = 0.0
        cos_img[visible <= 0] = 0.0
        # Per-view synthetic image: shift the RGB gradient so each view dumps
        # different colors (bake test would be trivial if all views had the
        # same image at the same world location).
        rng = np.random.default_rng(image_seed)
        tint = rng.uniform(0.2, 0.8, size=3).astype(np.float32)
        yy, xx = np.meshgrid(np.arange(args.himg), np.arange(args.himg),
                             indexing="ij")
        image = np.stack([
            (xx / args.himg).astype(np.float32) * tint[0],
            (yy / args.himg).astype(np.float32) * tint[1],
            (1.0 - (xx + yy).astype(np.float32) / (2 * args.himg)) * tint[2],
        ], -1)
        image *= visible[..., None]
        ref_tex, ref_cos = back_project_numpy(
            tex_pos, tex_cov, image, depth, visible, cos_img,
            w2c, proj00, proj11, depth_thres=3e-3)
        return w2c, image, depth, visible, cos_img, ref_tex, ref_cos

    if args.multiview:
        elevs = [0.0, 0.0, 0.0, 0.0]
        azims = [0.0, 90.0, 180.0, 270.0]
        per_view = []
        for vi, (e, a) in enumerate(zip(elevs, azims)):
            print(f"view {vi}: elev={e} azim={a}", file=sys.stderr)
            w2c, image, depth, visible, cos_img, ref_tex, ref_cos = \
                render_and_back_project(e, a, image_seed=1000 + vi)
            print(f"  visible={int(visible.sum())} written={int((ref_cos>0).sum())}",
                  file=sys.stderr)
            per_view.append((w2c, image, depth, visible, cos_img, ref_tex, ref_cos))

        # MeshRender.bake_texture / fast_bake_texture: weighted blend with
        # weight = cos^exp. Default weights[i] = 1.0, exp=6.
        Htex = args.htex; Wtex = args.htex
        tex_merge = np.zeros((Htex, Wtex, 3), dtype=np.float32)
        trust = np.zeros((Htex, Wtex), dtype=np.float32)
        for vi, (_, _, _, _, _, ref_tex, ref_cos) in enumerate(per_view):
            w_view = ref_cos ** args.bake_exp
            # The ref_cos>0 skip optimization in fast_bake_texture: skip if
            # >99% of view's painted texels are already painted. Reproducing
            # exactly so C-side can match.
            view_sum = (ref_cos > 0).sum()
            painted_sum = ((ref_cos > 0) & (trust > 0)).sum()
            if view_sum > 0 and painted_sum / view_sum > 0.99:
                print(f"  view {vi}: skipped (already 99% painted)",
                      file=sys.stderr)
                continue
            tex_merge += ref_tex * w_view[..., None]
            trust += w_view
        bake_tex = tex_merge / np.clip(trust, 1e-8, None)[..., None]
        bake_trust_mask = (trust > 1e-8).astype(np.float32)
        print(f"baked texels (trust>1e-8): {int(bake_trust_mask.sum())}",
              file=sys.stderr)

        np.save(os.path.join(args.outdir, "tex_pos.npy"), tex_pos)
        np.save(os.path.join(args.outdir, "tex_cov.npy"), tex_cov)
        for vi, (w2c, image, depth, visible, cos_img, ref_tex, ref_cos) in enumerate(per_view):
            np.save(os.path.join(args.outdir, f"view_{vi}_image.npy"), image)
            np.save(os.path.join(args.outdir, f"view_{vi}_depth.npy"), depth)
            np.save(os.path.join(args.outdir, f"view_{vi}_visible.npy"), visible)
            np.save(os.path.join(args.outdir, f"view_{vi}_cos.npy"), cos_img)
            np.save(os.path.join(args.outdir, f"view_{vi}_w2c.npy"), w2c.T.reshape(-1))
            np.save(os.path.join(args.outdir, f"view_{vi}_ref_tex.npy"), ref_tex)
            np.save(os.path.join(args.outdir, f"view_{vi}_ref_cos.npy"), ref_cos)
        np.save(os.path.join(args.outdir, "proj.npy"),
                np.array([proj00, proj11], dtype=np.float32))
        np.save(os.path.join(args.outdir, "bake_tex.npy"), bake_tex)
        np.save(os.path.join(args.outdir, "bake_trust.npy"), bake_trust_mask)
        with open(os.path.join(args.outdir, "meta.json"), "w") as f:
            json.dump({
                "Htex": Htex, "Wtex": Wtex,
                "Himg": args.himg, "Wimg": args.himg,
                "C": 3, "depth_thres": 3e-3,
                "n_views": len(per_view), "elevs": elevs, "azims": azims,
                "dist": args.dist, "ortho": args.ortho,
                "bake_exp": args.bake_exp,
            }, f, indent=2)
        print(f"wrote {args.outdir} (multi-view)", file=sys.stderr)
    else:
        print(f"render view elev={args.elev} azim={args.azim} dist={args.dist}",
              file=sys.stderr)
        w2c, image, depth, visible, cos_img, ref_tex, ref_cos = \
            render_and_back_project(args.elev, args.azim, image_seed=999)
        print(f"  written texels: {int((ref_cos > 0).sum())}", file=sys.stderr)
        np.save(os.path.join(args.outdir, "tex_pos.npy"), tex_pos)
        np.save(os.path.join(args.outdir, "tex_cov.npy"), tex_cov)
        np.save(os.path.join(args.outdir, "image.npy"),   image)
        np.save(os.path.join(args.outdir, "depth.npy"),   depth)
        np.save(os.path.join(args.outdir, "visible.npy"), visible)
        np.save(os.path.join(args.outdir, "cos.npy"),     cos_img)
        np.save(os.path.join(args.outdir, "w2c.npy"),     w2c.T.reshape(-1))
        np.save(os.path.join(args.outdir, "proj.npy"),
                np.array([proj00, proj11], dtype=np.float32))
        np.save(os.path.join(args.outdir, "ref_tex.npy"), ref_tex)
        np.save(os.path.join(args.outdir, "ref_cos.npy"), ref_cos)
        with open(os.path.join(args.outdir, "meta.json"), "w") as f:
            json.dump({
                "Htex": args.htex, "Wtex": args.htex,
                "Himg": args.himg, "Wimg": args.himg,
                "C": 3, "depth_thres": 3e-3,
                "elev": args.elev, "azim": args.azim, "dist": args.dist,
                "ortho": args.ortho,
            }, f, indent=2)
        print(f"wrote {args.outdir}", file=sys.stderr)


if __name__ == "__main__":
    main()
