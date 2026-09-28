"""Conservative cleanup of narrow Pixal3D protrusions around the eyes.

Below the lid, remove thin, supported foreground strips. Above it, reduce
local depth bulges along portrait rays, bounded by a measured back layer.
These restricted operations do not reconstruct the underlying lid anatomy.
"""
from __future__ import annotations

import warnings
import numpy as np

from .lids import eye_contour

NEIGHBOURHOOD_M = 0.0015       # radius of the median's footprint
MIN_RISE_M = 0.0008
MAX_RISE_M = 0.008            # reject discontinuities to unrelated surfaces
LOCAL_RADIUS_M = 0.028


def depth_map(positions, triangles, cam, origin, shape):
    """Nearest two-sided surface depth at integer portrait-pixel samples.

    Rasterize reciprocal camera Z for perspective-correct interpolation,
    then convert it to distance along the ray. Vertex splats are unsuitable:
    a sparse cheek mesh can otherwise expose a dense rear eye surface.
    Uncovered pixels have infinite depth. Geometry behind the camera is
    ignored (the fitted head lies entirely in front of it).
    """
    height, width = shape
    pix = cam.project(positions) - origin
    pp = pix[triangles]
    low = np.floor(pp.min(1)).astype(int)
    high = np.ceil(pp.max(1)).astype(int)
    z = positions[:, 2] - cam.origin[2]
    ids = np.flatnonzero((high[:, 0] >= 0) & (low[:, 0] < width) &
                         (high[:, 1] >= 0) & (low[:, 1] < height) & (z[triangles] > 0).all(1))
    field = np.full(shape, np.inf)
    for i in ids:
        lo = np.maximum(low[i], 0)
        hi = np.minimum(high[i], [width - 1, height - 1])
        yy, xx = np.mgrid[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        a, b, c = pp[i]
        den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(den) < 1e-10:
            continue
        u = ((b[1] - c[1]) * (xx - c[0]) + (c[0] - b[0]) * (yy - c[1])) / den
        v = ((c[1] - a[1]) * (xx - c[0]) + (a[0] - c[0]) * (yy - c[1])) / den
        w = 1 - u - v
        inv = u / z[triangles[i, 0]] + v / z[triangles[i, 1]] + w / z[triangles[i, 2]]
        depth = np.where((u >= -1e-6) & (v >= -1e-6) & (w >= -1e-6) & (inv > 0),
                         1 / np.maximum(inv, 1e-12), np.inf)
        patch = field[lo[1]:hi[1] + 1, lo[0]:hi[0] + 1]
        np.minimum(patch, depth, out=patch)
    yy, xx = np.mgrid[:height, :width]
    _, rays = cam.rays(xx + origin[0], yy + origin[1])
    return field / rays[..., 2]


def raised_pixels(depth, radius, min_rise, max_rise):
    """Return the median, raised-pixel mask, and reliable-support mask."""
    field = np.where(np.isfinite(depth), depth, np.nan)
    size = 2 * radius + 1
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(field, radius, constant_values=np.nan), (size, size))
    median = np.empty_like(field)
    supported = np.zeros(field.shape, bool)
    # nanmedian copies its input; bound that workspace for large portraits.
    for row in range(0, len(field), 32):
        block = windows[row:row + 32]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-empty windows
            median[row:row + 32] = np.nanmedian(block, axis=(-2, -1))
        supported[row:row + 32] = np.isfinite(block).sum(axis=(-2, -1)) >= 0.9 * size * size
    rise = median - field
    return median, supported & (rise > min_rise) & (rise < max_rise), supported


def remove_slivers(mesh, keep, cam, eyes, poses):
    """Remove only triangles whose vertices and centre lie on a supported sliver.

    Restrict edits below the lower lid, outside 1.8x the fissure contour,
    and within 28 mm of the eye. The wet margin and upper lid are untouched.
    Return a new keep mask and per-eye counts; never move vertices or UVs.
    """
    pos, tris = mesh["positions"], mesh["triangles"]
    result = keep.copy()
    info = {}
    # Reference the same input surface for both eyes (no order dependence).
    reference_tris = tris[keep]
    for eye, pose in zip(eyes, poses):
        contour = eye_contour(eye)
        ids = np.flatnonzero(np.linalg.norm(pos - pose.center, axis=1) < LOCAL_RADIUS_M * pose.units_per_m)
        if contour is None or not len(ids) or not len(reference_tris):
            info[eye.side] = {"triangles_removed": 0}
            continue
        pix = cam.project(pos[ids])
        distance = float(np.linalg.norm(pose.center - cam.origin))
        px_per_m = cam.focal * pose.units_per_m / (distance * cam.scale)
        radius = max(2, min(12, round(NEIGHBOURHOOD_M * px_per_m)))
        origin = np.floor(pix.min(0)).astype(int) - radius - 1
        xy = np.rint(pix - origin).astype(int)
        width, height = xy.max(0) + radius + 2
        depth = depth_map(pos, reference_tris, cam, origin, (height, width))
        median, _, supported = raised_pixels(depth, radius, MIN_RISE_M * pose.units_per_m,
                                       MAX_RISE_M * pose.units_per_m)
        ref = median[xy[:, 1], xy[:, 0]]
        rise = ref - np.linalg.norm(pos[ids] - cam.origin, axis=1)
        angle = np.arctan2(pix[:, 1] - eye.cy, pix[:, 0] - eye.cx)
        q = np.hypot(pix[:, 0] - eye.cx, pix[:, 1] - eye.cy) / np.maximum(contour(angle), 1e-6)
        bad = np.zeros(len(pos), bool)
        # Evaluate the actual vertex depth against the reference, not the
        # rasterized foreground sample: subpixel strips can fall between
        # pixels and must not survive as intermittent fragments.
        bad[ids] = (supported[xy[:, 1], xy[:, 0]] & (rise > MIN_RISE_M * pose.units_per_m) &
                    (rise < MAX_RISE_M * pose.units_per_m) & (q > 1.8) & (pix[:, 1] > eye.cy + eye.r))
        candidates = np.flatnonzero(result & bad[tris].all(1))
        # Coarse broad patches may have raised corners but a well-supported
        # interior; a vertex-only decision would wrongly remove the whole face.
        centres = pos[tris[candidates]].mean(1)
        cc = np.rint(cam.project(centres) - origin).astype(int)
        inside = (cc[:, 0] >= 0) & (cc[:, 0] < width) & (cc[:, 1] >= 0) & (cc[:, 1] < height)
        cc = cc[inside]
        centre_rise = median[cc[:, 1], cc[:, 0]] - np.linalg.norm(centres[inside] - cam.origin, axis=1)
        drop = candidates[inside][supported[cc[:, 1], cc[:, 0]] &
                                  (centre_rise > MIN_RISE_M * pose.units_per_m) &
                                  (centre_rise < MAX_RISE_M * pose.units_per_m)]
        result[drop] = False
        info[eye.side] = {"triangles_removed": int(len(drop)), "median_radius_px": radius}
    return result, info


def _sample_finite(field, xy):
    """Bilinear samples requiring four finite neighbours; missing is NaN."""
    x, y = xy.T
    h, w = field.shape
    ix = np.clip(np.floor(x).astype(int), 0, w - 2)
    iy = np.clip(np.floor(y).astype(int), 0, h - 2)
    fx, fy = x - ix, y - iy
    values = np.stack([field[iy, ix], field[iy, ix + 1], field[iy + 1, ix], field[iy + 1, ix + 1]], 1)
    weights = np.stack([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy], 1)
    valid = np.isfinite(values).all(1) & (x >= 0) & (y >= 0) & (x <= w - 1) & (y <= h - 1)
    result = (np.where(np.isfinite(values), values, 0) * weights).sum(1)
    return np.where(valid, result, np.nan)


def _smooth_offsets(pos, triangles, offsets, units_per_m):
    """Diffuse offsets on front-facing, geometrically welded adjacency.

    Zero candidates stay fixed. This reduces abrupt displacement changes
    without spreading a cleanup into the brow, cut margin, or back layer.
    The caller reapplies the measured rear-surface bound afterwards.
    """
    vertices = np.unique(triangles)
    if not len(vertices):
        return offsets
    _, first, groups = np.unique(np.rint(pos[vertices] / 1e-8).astype(np.int64),
                                 axis=0, return_index=True, return_inverse=True)
    faces = groups[np.searchsorted(vertices, triangles)]
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges.sort(1)
    edges = np.unique(edges, axis=0)
    edges = edges[edges[:, 0] != edges[:, 1]]
    points = pos[vertices[first]]
    weights = 1 / np.maximum(np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1),
                              .0002 * units_per_m)
    values = offsets[vertices[first]].copy()
    movable = values > 0
    denom = (np.bincount(edges[:, 0], weights, minlength=len(points)) +
             np.bincount(edges[:, 1], weights, minlength=len(points)))
    for _ in range(6):
        total = (np.bincount(edges[:, 0], weights * values[edges[:, 1]], minlength=len(points)) +
                 np.bincount(edges[:, 1], weights * values[edges[:, 0]], minlength=len(points)))
        values = np.where(movable, .5 * values + .5 * total / np.maximum(denom, 1e-12), 0)
    result = offsets.copy()
    result[vertices] = values[groups]
    return result


def relax_upper_lids(mesh, keep, cam, eyes, poses):
    """Reduce narrow raised upper-lid bulges without removing geometry.

    Work after lid draping, leaving its margin and wet meshes fixed. Move
    only along portrait rays toward a locally supported median depth. A
    measured back-facing layer bounds each move, retaining 0.2 mm of
    clearance. No back layer or insufficient support means no change.
    This reduces local bulges; it does not reconstruct the lid anatomy or
    remove broad overhangs. Maximum movement is 3.5 mm.
    """
    from .lids import vertex_normals
    pos = mesh["positions"].copy()
    triangles = mesh["triangles"][keep]
    changed = np.zeros(len(pos), bool)
    info = {}
    for eye, pose in zip(eyes, poses):
        stats = {"moved_vertices": 0, "max_move_mm": 0., "mean_move_mm": 0.}
        info[eye.side] = stats
        contour = eye_contour(eye)
        if contour is None or not len(triangles):
            continue
        k = pose.units_per_m
        pix = cam.project(pos)
        theta = np.arctan2(pix[:, 1] - eye.cy, pix[:, 0] - eye.cx)
        q = np.hypot(pix[:, 0] - eye.cx, pix[:, 1] - eye.cy) / np.maximum(contour(theta), 1e-6)
        up = (eye.cy - pix[:, 1]) / eye.r
        select = ((q > 1.05) & (q < 2.2) & (up > .55) & (up < 1.7) &
                  (np.linalg.norm(pos - pose.center, axis=1) < .022 * k))
        ids = np.flatnonzero(select)
        if not len(ids):
            continue
        px_per_m = cam.focal * k / (np.linalg.norm(pose.center - cam.origin) * cam.scale)
        radius = max(2, min(12, round(.0015 * px_per_m)))
        origin = np.floor(pix[ids].min(0)).astype(int) - radius - 2
        width, height = np.ceil(pix[ids].max(0) - origin).astype(int) + radius + 3
        local = np.linalg.norm(pos[triangles].mean(1) - pose.center, axis=1) < .028 * k
        local_tri = triangles[local]
        a, b, c = (pos[local_tri[:, j]] for j in range(3))
        front = (np.cross(b - a, c - a) * (cam.origin - (a + b + c) / 3)).sum(1) > 0
        if not front.any() or front.all():
            continue
        depth = depth_map(pos, local_tri[front], cam, origin, (height, width))
        back = depth_map(pos, local_tri[~front], cam, origin, (height, width))
        median, _, support = raised_pixels(depth, radius, .0003 * k, .006 * k)
        xy = pix[ids] - origin
        ref, rear = _sample_finite(median, xy), _sample_finite(back, xy)
        supported = _sample_finite(support.astype(float), xy) > .99
        dist = np.linalg.norm(pos[ids] - cam.origin, axis=1)
        rise = (ref - dist) / k
        taper = np.minimum.reduce([np.clip((q[ids] - 1.05) / .2, 0, 1),
                                    np.clip((2.2 - q[ids]) / .4, 0, 1),
                                    np.clip((up[ids] - .55) / .2, 0, 1),
                                    np.clip((1.7 - up[ids]) / .25, 0, 1)])
        amount = np.minimum(np.maximum(ref - dist, 0), .0035 * k)
        amount *= np.clip((rise - .0003) / .0006, 0, 1) * taper
        valid = (supported & np.isfinite(ref) & np.isfinite(rear) &
                 (rear > dist + .0002 * k) & (rise > 0) & (rise < .006))
        clearance = np.maximum(rear - dist - .0002 * k, 0)
        full = np.zeros(len(pos))
        full[ids] = np.where(valid, np.minimum(amount, clearance), 0)
        full = _smooth_offsets(pos, local_tri[front], full, k)
        amount = np.where(valid, np.minimum(full[ids], clearance), 0)
        move = amount > .00001 * k
        ii = ids[move]
        pos[ii] += (pos[ii] - cam.origin) / dist[move, None] * amount[move, None]
        changed[ii] = True
        if move.any():
            stats.update(moved_vertices=int(move.sum()), max_move_mm=float(amount[move].max() / k * 1000),
                         mean_move_mm=float(amount[move].mean() / k * 1000))
    normals = mesh["normals"].copy()
    if changed.any():
        affected = np.zeros(len(pos), bool)
        affected[triangles[changed[triangles].any(1)].ravel()] = True
        # Include unchanged seam copies adjacent on the other UV chart.
        _, groups = np.unique(np.rint(pos / 1e-8).astype(np.int64), axis=0, return_inverse=True)
        shared = np.zeros(groups.max() + 1, bool)
        np.logical_or.at(shared, groups, affected)
        affected = shared[groups]
        fresh = vertex_normals(pos, triangles)
        good = affected & (np.linalg.norm(fresh, axis=1) > 0)
        normals[good] = fresh[good]
    return dict(mesh, positions=pos, normals=normals), info
