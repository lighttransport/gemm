"""Minimal software rasterizer to replace nvdiffrast.torch for TRELLIS.2 UV texturing.

The texturing pipeline calls only:
  - dr.RasterizeCudaContext()
  - dr.rasterize(ctx, pos [1,V,4], tri [F,3], resolution=[H,W])
  - dr.interpolate(attrs [1,V,C], rast, tri)

Input pos is 2D UV in clip space ([-1,1], z=0, w=1), so this is 2D triangle rasterization.
UV unwrap produces non-overlapping charts, so no depth test is needed.
"""

import torch


class RasterizeCudaContext:
    def __init__(self, device=None):
        self.device = device


def _edge(ax, ay, bx, by, px, py):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def rasterize(ctx, pos, tri, resolution):
    """Return (rast[B,H,W,4], rast_db) matching nvdiffrast contract.

    rast[...,0:2] = barycentric (u, v) wrt verts 1,2 (v0 has w0 = 1-u-v).
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
    py = (1.0 - ndc[:, 1]) * 0.5 * H  # flip y to image coords

    tri_i = tri.to(torch.long)
    F = tri_i.shape[0]

    rast = torch.zeros((1, H, W, 4), dtype=torch.float32, device=device)

    px_c = px.cpu().numpy()
    py_c = py.cpu().numpy()
    zow_c = zow.cpu().numpy()
    tri_c = tri_i.cpu().numpy()

    # allocate on CPU then copy; triangles are small and sparse in image.
    import numpy as np
    rast_np = np.zeros((H, W, 4), dtype=np.float32)

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

        u = w1  # bary wrt v1
        v = w2  # bary wrt v2
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
    out = w.unsqueeze(-1) * a0 + u.unsqueeze(-1) * a1 + v.unsqueeze(-1) * a2
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
