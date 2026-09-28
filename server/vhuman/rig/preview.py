"""A small software rasterizer for rig previews and tests (no GPU, no GL):
flat or smooth Lambert shading with a z-buffer, orthographic or perspective."""
from __future__ import annotations

import math

import numpy as np

from .common import normalize, rotation, vertex_normals


def view_matrix(yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> np.ndarray:
    """Camera rotation looking at the face (+Z towards the viewer) after
    turning the head by yaw (about Y) and pitch (about X)."""
    return rotation([1, 0, 0], math.radians(pitch_deg)) @ rotation([0, 1, 0], math.radians(yaw_deg))


def render(pos, tris, size=512, yaw=0.0, pitch=0.0, center=None, extent=None, colors=None,
           tri_colors=None, uv=None, tri_uv=None, texture=None, wire=False, bg=(40, 42, 48)) -> np.ndarray:
    """RGB uint8 image of one mesh (see render_scene)."""
    return render_scene([dict(pos=pos, tris=tris, colors=colors, tri_colors=tri_colors, uv=uv, tri_uv=tri_uv,
                              texture=texture)], size, yaw, pitch, center, extent, wire, bg)


def render_scene(meshes, size=512, yaw=0.0, pitch=0.0, center=None, extent=None, wire=False,
                 bg=(40, 42, 48)) -> np.ndarray:
    """Meshes sharing one z-buffer; each: pos (V,3), tris, and optionally
    colors (V,3), tri_colors (T,3), or uv + tri_uv + texture (glTF UVs).
    Orthographic along -Z after the view rotation; `extent` is the half-size
    of the view in metres."""
    R = view_matrix(yaw, pitch)
    allp = np.concatenate([np.asarray(m["pos"], np.float64) for m in meshes])
    c = allp[np.isfinite(allp).all(1)].mean(0) if center is None else np.asarray(center, np.float64)
    if extent is None:
        q = (allp - c) @ R.T
        extent = float(np.abs(q[np.isfinite(q).all(1), :2]).max()) * 1.05
    s = size / (2 * extent)
    light = normalize(np.array([0.3, 0.4, 1.0]))
    img = np.zeros((size, size, 3))
    img[:] = np.asarray(bg) / 255
    zb = np.full((size, size), -np.inf)
    for mesh in meshes:
        p = np.asarray(mesh["pos"], np.float64)
        tris = np.asarray(mesh["tris"])
        q = (p - c) @ R.T
        xy = np.stack([q[:, 0] * s + size / 2, size / 2 - q[:, 1] * s], 1)
        z = q[:, 2]
        n = vertex_normals(q, tris)
        fn = np.cross(q[tris[:, 1]] - q[tris[:, 0]], q[tris[:, 2]] - q[tris[:, 0]])
        tex = mesh.get("texture")
        tex = None if tex is None else np.asarray(tex, np.float64) / 255
        colors, tri_colors, uv, tri_uv = mesh.get("colors"), mesh.get("tri_colors"), mesh.get("uv"), mesh.get("tri_uv")
        for ti in np.flatnonzero(fn[:, 2] > 0):
            a, b, cc = tris[ti]
            pa, pb, pc = xy[a], xy[b], xy[cc]
            if not (np.isfinite(pa).all() and np.isfinite(pb).all() and np.isfinite(pc).all()):
                continue
            lo = np.maximum(np.floor(np.minimum(np.minimum(pa, pb), pc)).astype(int), 0)
            hi = np.minimum(np.ceil(np.maximum(np.maximum(pa, pb), pc)).astype(int), size - 1)
            if (hi < lo).any():
                continue
            yy, xx = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
            den = (pb[1] - pc[1]) * (pa[0] - pc[0]) + (pc[0] - pb[0]) * (pa[1] - pc[1])
            if abs(den) < 1e-12:
                continue
            w0 = ((pb[1] - pc[1]) * (xx + .5 - pc[0]) + (pc[0] - pb[0]) * (yy + .5 - pc[1])) / den
            w1 = ((pc[1] - pa[1]) * (xx + .5 - pc[0]) + (pa[0] - pc[0]) * (yy + .5 - pc[1])) / den
            w2 = 1 - w0 - w1
            m = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
            if not m.any():
                continue
            zz = w0 * z[a] + w1 * z[b] + w2 * z[cc]
            sub = zb[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
            m &= zz > sub
            if not m.any():
                continue
            sub[m] = zz[m]
            nn = normalize(w0[m, None] * n[a] + w1[m, None] * n[b] + w2[m, None] * n[cc])
            shade = 0.25 + 0.75 * np.clip(nn @ light, 0, 1)
            if tex is not None and tri_uv is not None:
                ua, ub, uc = uv[tri_uv[ti]]
                u = w0[m, None] * ua + w1[m, None] * ub + w2[m, None] * uc
                h, w = tex.shape[:2]
                px = np.clip((u[:, 0] * (w - 1)).astype(int), 0, w - 1)
                py = np.clip((u[:, 1] * (h - 1)).astype(int), 0, h - 1)      # glTF: v down
                col = tex[py, px, :3] * (0.55 + 0.45 * shade[:, None])
            elif colors is not None:
                cv = np.asarray(colors, np.float64)
                col = (w0[m, None] * cv[a] + w1[m, None] * cv[b] + w2[m, None] * cv[cc]) * shade[:, None]
            elif tri_colors is not None:
                col = np.asarray(tri_colors[ti], np.float64)[None] * shade[:, None]
            else:
                col = np.repeat(shade[:, None], 3, 1) * np.array([0.85, 0.78, 0.72])
            patch = img[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
            patch[m] = col
            if wire:
                edge = m & (np.minimum(np.minimum(w0, w1), w2) < 0.04)
                patch[edge] = patch[edge] * 0.4
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)
