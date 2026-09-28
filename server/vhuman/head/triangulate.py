"""Checked planar annulus triangulation for experimental lid reconstruction.

Not enabled by the production fitter. Source boundaries must be simple and
nested before triangulation; a closed 3-D loop is not sufficient. The optional
mapbox-earcut dependency is imported only after input validation.
"""
from __future__ import annotations

import numpy as np


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _intersections(a, b, tolerance, same=False):
    a0, a1 = a[:, None], np.roll(a, -1, axis=0)[:, None]
    b0, b1 = b[None, :], np.roll(b, -1, axis=0)[None, :]
    scale = max(float(np.ptp(np.concatenate([a, b]), axis=0).max()), 1e-12)
    epsilon = tolerance * scale
    def sign(v):
        return np.where(v > epsilon, 1, np.where(v < -epsilon, -1, 0))
    s1, s2 = sign(_cross(a1 - a0, b0 - a0)), sign(_cross(a1 - a0, b1 - a0))
    s3, s4 = sign(_cross(b1 - b0, a0 - b0)), sign(_cross(b1 - b0, a1 - b0))
    overlap = ((np.minimum(a0, a1) <= np.maximum(b0, b1) + tolerance) &
               (np.minimum(b0, b1) <= np.maximum(a0, a1) + tolerance)).all(-1)
    hit = (s1 * s2 <= 0) & (s3 * s4 <= 0) & overlap
    if same:
        i = np.arange(len(a))
        hit[i, i] = False
        hit[i, (i + 1) % len(a)] = False
        hit[i, (i - 1) % len(a)] = False
    return bool(hit.any())


def _area(p):
    return float(_cross(p, np.roll(p, -1, axis=0)).sum() * .5)


def _inside(point, polygon):
    a, b = polygon, np.roll(polygon, -1, axis=0)
    crossing = (a[:, 1] > point[1]) != (b[:, 1] > point[1])
    a, b = a[crossing], b[crossing]
    x = a[:, 0] + (point[1] - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return bool(np.count_nonzero(x > point[0]) % 2)


def triangulate_annulus(outer, inner):
    """Return CCW triangle indices into concatenated [outer, inner] vertices.

    Preserve every boundary edge. Reject crossings, touching loops, collapsed
    edges, non-nesting, inconsistent winding, and area/coverage mismatches.
    Coordinates may be portrait pixels or any other consistent planar units.
    """
    rings = [np.asarray(p, dtype=np.float64) for p in (outer, inner)]
    for p in rings:
        if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
            raise ValueError('rings need at least three finite 2-D points')
    points = np.concatenate(rings)
    # Translate near the origin for stable area sums with large image offsets.
    offset = points.mean(0)
    rings = [p - offset for p in rings]
    points = np.concatenate(rings)
    scale = max(float(np.ptp(points, axis=0).max()), 1e-12)
    tolerance = scale * 1e-10
    for p in rings:
        if (np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1) <= tolerance).any():
            raise ValueError('collapsed boundary edge')
        if _intersections(p, p, tolerance, same=True):
            raise ValueError('self-intersecting boundary')
        if abs(_area(p)) <= tolerance * scale:
            raise ValueError('zero-area boundary')
    if _intersections(*rings, tolerance):
        raise ValueError('boundaries intersect or touch')
    if not _inside(rings[1][0], rings[0]):
        raise ValueError('inner boundary is not inside outer boundary')
    try:
        import mapbox_earcut
    except ImportError as exc:
        raise RuntimeError('install server/vhuman/requirements-remesh.txt to triangulate lid patches') from exc
    ends = np.array([len(rings[0]), len(points)], np.uint32)
    tri = mapbox_earcut.triangulate_float64(np.ascontiguousarray(points), ends).reshape(-1, 3)
    if not len(tri):
        raise ValueError('empty triangulation')
    vertices = points[tri]
    areas = _cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]) * .5
    if (areas < 0).all():
        tri = tri[:, [0, 2, 1]]
        areas = -areas
    if (areas <= tolerance * scale).any():
        raise ValueError('degenerate or inconsistently oriented triangles')
    expected_area = abs(_area(rings[0])) - abs(_area(rings[1]))
    if not np.isclose(areas.sum(), expected_area, rtol=1e-8, atol=tolerance * scale):
        raise ValueError('triangulation does not cover the annulus')
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    unique, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    direction = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    if (counts > 2).any() or ((counts == 2) & (direction != 0)).any():
        raise ValueError('nonmanifold or inconsistently oriented edges')
    expected = set()
    base = 0
    for ring in rings:
        expected.update(tuple(sorted((base + i, base + (i + 1) % len(ring)))) for i in range(len(ring)))
        base += len(ring)
    if {tuple(e) for e in unique[counts == 1]} != expected:
        raise ValueError('triangulation changed boundary edges')
    return tri
