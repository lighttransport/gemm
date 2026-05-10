"""Minimal software rasterizer to replace nvdiffrast.torch for TRELLIS.2 UV texturing.

The texturing pipeline calls only:
  - dr.RasterizeCudaContext()
  - dr.rasterize(ctx, pos [1,V,4], tri [F,3], resolution=[H,W])
  - dr.interpolate(attrs [1,V,C], rast, tri)

Input pos is 2D UV in clip space ([-1,1], z=0, w=1), so this is 2D triangle rasterization.
UV unwrap produces non-overlapping charts, so no depth test is needed.
"""

import numpy as np
import torch

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


class RasterizeCudaContext:
    def __init__(self, device=None):
        self.device = device


def _edge(ax, ay, bx, by, px, py):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


if _HAVE_NUMBA:
    @njit(cache=True, parallel=True, fastmath=True)
    def _rasterize_numba(tri_c, px_c, py_c, zow_c, H, W, rast_np):
        F = tri_c.shape[0]
        for f in prange(F):
            i0 = tri_c[f, 0]
            i1 = tri_c[f, 1]
            i2 = tri_c[f, 2]
            ax = px_c[i0]; ay = py_c[i0]
            bx = px_c[i1]; by = py_c[i1]
            cx = px_c[i2]; cy = py_c[i2]
            z0 = zow_c[i0]; z1 = zow_c[i1]; z2 = zow_c[i2]

            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if area == 0.0:
                continue
            inv_area = 1.0 / area

            xmin_f = ax if ax < bx else bx
            if cx < xmin_f:
                xmin_f = cx
            xmax_f = ax if ax > bx else bx
            if cx > xmax_f:
                xmax_f = cx
            ymin_f = ay if ay < by else by
            if cy < ymin_f:
                ymin_f = cy
            ymax_f = ay if ay > by else by
            if cy > ymax_f:
                ymax_f = cy

            xmin = int(np.floor(xmin_f))
            if xmin < 0:
                xmin = 0
            xmax = int(np.ceil(xmax_f))
            if xmax > W - 1:
                xmax = W - 1
            ymin = int(np.floor(ymin_f))
            if ymin < 0:
                ymin = 0
            ymax = int(np.ceil(ymax_f))
            if ymax > H - 1:
                ymax = H - 1
            if xmax < xmin or ymax < ymin:
                continue

            tid = np.float32(f + 1)
            for y in range(ymin, ymax + 1):
                py = np.float32(y) + np.float32(0.5)
                for x in range(xmin, xmax + 1):
                    px = np.float32(x) + np.float32(0.5)
                    w0 = ((bx - px) * (cy - py) - (by - py) * (cx - px)) * inv_area
                    w1 = ((cx - px) * (ay - py) - (cy - py) * (ax - px)) * inv_area
                    w2 = 1.0 - w0 - w1
                    if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0:
                        rast_np[y, x, 0] = w0
                        rast_np[y, x, 1] = w1
                        rast_np[y, x, 2] = w0 * z0 + w1 * z1 + w2 * z2
                        rast_np[y, x, 3] = tid


def rasterize(ctx, pos, tri, resolution):
    """Return (rast[B,H,W,4], rast_db) matching nvdiffrast contract.

    rast[...,0:2] = barycentric (u, v) wrt verts 0,1 (v2 has w = 1-u-v).
    rast[...,2]   = z/w.
    rast[...,3]   = triangle_id + 1 (0 = empty pixel).
    """
    assert pos.dim() == 3 and pos.shape[0] == 1 and pos.shape[-1] == 4
    H, W = int(resolution[0]), int(resolution[1])
    device = pos.device

    p = pos[0]
    ndc = p[:, :2] / p[:, 3:4].clamp_min(1e-20)
    zow = (p[:, 2] / p[:, 3].clamp_min(1e-20))
    # map NDC [-1,1] to pixel centre coords [0, W] / [0, H]. nvdiffrast: x right, y up.
    px = (ndc[:, 0] + 1.0) * 0.5 * W
    py = (ndc[:, 1] + 1.0) * 0.5 * H  # nvdiffrast keeps NDC y orientation

    tri_i = tri.to(torch.long)
    F = tri_i.shape[0]

    rast = torch.zeros((1, H, W, 4), dtype=torch.float32, device=device)

    px_c = px.cpu().numpy().astype(np.float32)
    py_c = py.cpu().numpy().astype(np.float32)
    zow_c = zow.cpu().numpy().astype(np.float32)
    tri_c = tri_i.cpu().numpy().astype(np.int64)

    rast_np = np.zeros((H, W, 4), dtype=np.float32)

    if _HAVE_NUMBA:
        _rasterize_numba(tri_c, px_c, py_c, zow_c, H, W, rast_np)
        rast[0] = torch.from_numpy(rast_np).to(device)
        return rast, None

    for f in range(F):
        i0, i1, i2 = tri_c[f]
        ax, ay = px_c[i0], py_c[i0]
        bx, by = px_c[i1], py_c[i1]
        cx, cy = px_c[i2], py_c[i2]
        z0, z1, z2 = zow_c[i0], zow_c[i1], zow_c[i2]

        area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
        if area == 0.0:
            continue
        inv_area = 1.0 / area

        xmin = max(0, int(np.floor(min(ax, bx, cx))))
        xmax = min(W - 1, int(np.ceil(max(ax, bx, cx))))
        ymin = max(0, int(np.floor(min(ay, by, cy))))
        ymax = min(H - 1, int(np.ceil(max(ay, by, cy))))
        if xmax < xmin or ymax < ymin:
            continue

        xs = np.arange(xmin, xmax + 1, dtype=np.float32) + 0.5
        ys = np.arange(ymin, ymax + 1, dtype=np.float32) + 0.5
        XX, YY = np.meshgrid(xs, ys)

        # edge functions: w0 opposite vertex 0, etc.
        w0 = ((bx - XX) * (cy - YY) - (by - YY) * (cx - XX)) * inv_area
        w1 = ((cx - XX) * (ay - YY) - (cy - YY) * (ax - XX)) * inv_area
        w2 = 1.0 - w0 - w1

        mask = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not mask.any():
            continue

        u = w0  # bary wrt v0
        v = w1  # bary wrt v1
        z = w0 * z0 + w1 * z1 + w2 * z2

        tile = rast_np[ymin:ymax + 1, xmin:xmax + 1]
        m = mask
        tile[m, 0] = u[m]
        tile[m, 1] = v[m]
        tile[m, 2] = z[m]
        tile[m, 3] = float(f + 1)

    rast[0] = torch.from_numpy(rast_np).to(device)
    return rast, None


def interpolate(attrs, rast, tri):
    """Return (out[B,H,W,C], out_db). Samples vertex attrs with rast barycentrics."""
    assert attrs.dim() == 3 and attrs.shape[0] == 1
    B, H, W, _ = rast.shape
    C = attrs.shape[-1]
    device = rast.device

    tri_i = tri.to(torch.long)
    u = rast[..., 0]
    v = rast[..., 1]
    w = 1.0 - u - v
    tid = rast[..., 3].to(torch.long)
    valid = tid > 0
    t_idx = (tid - 1).clamp_min(0)

    verts = tri_i[t_idx]  # [B,H,W,3]
    a0 = attrs[0][verts[..., 0]]
    a1 = attrs[0][verts[..., 1]]
    a2 = attrs[0][verts[..., 2]]
    # rast layout matches CUDA dump: pos = u*v0 + v*v1 + (1-u-v)*v2
    out = u.unsqueeze(-1) * a0 + v.unsqueeze(-1) * a1 + w.unsqueeze(-1) * a2
    out = out * valid.unsqueeze(-1).to(out.dtype)
    return out.reshape(B, H, W, C), None


def install_as_nvdiffrast():
    """Register this module as nvdiffrast.torch so `import nvdiffrast.torch as dr` picks it up."""
    import sys
    import types
    pkg = types.ModuleType('nvdiffrast')
    mod = types.ModuleType('nvdiffrast.torch')
    mod.RasterizeCudaContext = RasterizeCudaContext
    mod.rasterize = rasterize
    mod.interpolate = interpolate
    pkg.torch = mod
    sys.modules.setdefault('nvdiffrast', pkg)
    sys.modules['nvdiffrast.torch'] = mod
