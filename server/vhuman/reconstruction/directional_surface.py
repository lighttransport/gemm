"""Constrain scalar surface offsets along one fixed direction.

Oriented triangle area is affine in these offsets: the quadratic cross term
vanishes because both displaced edges are parallel to the same direction.
This preserves orientation relative to supplied reference frames, not global
injectivity, self-intersection freedom, or arbitrary animation poses.
"""
import numpy as np
from scipy.sparse import coo_matrix


def area_constraints(frames, triangles, direction):
    frames = np.asarray(frames, dtype=float)
    triangles = np.asarray(triangles)
    direction = np.asarray(direction, dtype=float)
    if (frames.ndim != 3 or frames.shape[-1] != 3 or not np.isfinite(frames).all()
            or triangles.ndim != 2 or triangles.shape[1] != 3
            or triangles.dtype.kind not in 'iu' or not len(triangles)
            or triangles.min() < 0 or triangles.max() >= frames.shape[1]
            or direction.shape != (3,) or not np.isfinite(direction).all()
            or not np.isclose(np.linalg.norm(direction), 1)):
        raise ValueError('invalid directional area inputs')
    values, columns = [], []
    for vertices in frames:
        p = vertices[triangles]
        e1, e2 = p[:,1]-p[:,0], p[:,2]-p[:,0]
        normal = np.cross(e1, e2)
        norm2 = np.square(normal).sum(1)
        valid = norm2 > 1e-24
        # Input offsets are millimetres; geometry is in metres.
        b = (np.cross(direction, e2[valid])*normal[valid]).sum(1)/norm2[valid]*.001
        c = (np.cross(e1[valid], direction)*normal[valid]).sum(1)/norm2[valid]*.001
        values.append(np.stack((-b-c, b, c), axis=1))
        columns.append(triangles[valid])
    values, columns = np.concatenate(values), np.concatenate(columns)
    if not len(values):
        raise ValueError('no nondegenerate reference triangles')
    rows = np.repeat(np.arange(len(values)), 3)
    return coo_matrix((values.ravel(), (rows, columns.ravel())),
                      shape=(len(values), frames.shape[1])).tocsr()


def constrain_offsets(frames, triangles, direction, desired_mm, *, minimum_ratio=.2,
                      limit_mm=2., tolerance=1e-8, max_sweeps=2000):
    """Project onto area halfspaces and a displacement box using Dykstra sweeps.

    Return offsets plus convergence diagnostics. Callers must reject a report
    with converged=False; this routine never writes or promotes a candidate.
    """
    if (not 0 < minimum_ratio < 1 or not np.isfinite(limit_mm) or limit_mm <= 0
            or not np.isfinite(tolerance) or tolerance <= 0 or max_sweeps < 1):
        raise ValueError('invalid directional surface constraints')
    matrix = area_constraints(frames, triangles, direction)
    desired = np.asarray(desired_mm, dtype=float)
    if desired.shape != (matrix.shape[1],) or not np.isfinite(desired).all():
        raise ValueError('invalid desired offsets')
    offsets = desired.copy()
    multipliers = np.zeros(matrix.shape[0])
    box_residual = np.zeros_like(offsets)
    active = set()
    bound = minimum_ratio-1
    converged = False
    for sweep in range(max_sweeps):
        before = offsets.copy()
        active.update(np.flatnonzero(matrix@offsets < bound-tolerance).tolist())
        for row in sorted(active):
            start, end = matrix.indptr[row:row+2]
            ids, values = matrix.indices[start:end], matrix.data[start:end]
            norm2 = float(values@values)
            if norm2 < 1e-30:
                continue
            # Undo this halfspace's previous projection before reprojecting.
            point = offsets[ids]-multipliers[row]*values
            multiplier = max(0., (bound-float(values@point))/norm2)
            offsets[ids] = point+multiplier*values
            multipliers[row] = multiplier
        point = offsets+box_residual
        offsets = np.clip(point, -limit_mm, limit_mm)
        box_residual = point-offsets
        ratios = 1+matrix@offsets
        if ratios.min() >= minimum_ratio-tolerance and np.max(np.abs(offsets-before)) < tolerance:
            converged = True
            break
    report = dict(converged=converged, sweeps=sweep+1, active_constraints=len(active),
                  minimum_area_ratio=float(ratios.min()), minimum_required=minimum_ratio,
                  max_offset_mm=float(np.abs(offsets).max()), limit_mm=limit_mm,
                  max_projection_correction_mm=float(np.abs(offsets-desired).max()))
    return offsets, report
