"""Constrain scalar surface offsets along one fixed direction.

Oriented triangle area is affine in these offsets: the quadratic cross term
vanishes because both displaced edges are parallel to the same direction.
This preserves orientation relative to supplied reference frames, not global
injectivity, self-intersection freedom, or arbitrary animation poses.
"""
import numpy as np
from scipy.sparse import coo_matrix, vstack


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
                      limit_mm=2., tolerance=1e-8, max_sweeps=2000, fixed_attachments=None,
                      maximum_edge_ratio=None):
    """Project onto area halfspaces and a displacement box using Dykstra sweeps.

    Optional (vertex_ids, barycentric_weights) attachments have zero displacement.
    Optional maximum_edge_ratio bounds edge stretch in every supplied pose;
    the scalar interval intersections are exact for the fixed direction.
    Return offsets plus convergence diagnostics. Callers must reject a report
    with converged=False; this routine never writes or promotes a candidate.
    """
    if (not 0 < minimum_ratio < 1 or not np.isfinite(limit_mm) or limit_mm <= 0
            or not np.isfinite(tolerance) or tolerance <= 0 or max_sweeps < 1):
        raise ValueError('invalid directional surface constraints')
    matrix = area_constraints(frames, triangles, direction)
    area_count = matrix.shape[0]
    bounds = np.full(area_count, minimum_ratio-1)
    edges = None
    if maximum_edge_ratio is not None:
        if not np.isfinite(maximum_edge_ratio) or maximum_edge_ratio <= 1:
            raise ValueError('maximum edge ratio must exceed one')
        frames, triangles, direction = np.asarray(frames), np.asarray(triangles), np.asarray(direction)
        edges = np.unique(np.sort(triangles[:, [[0,1], [1,2], [2,0]]].reshape(-1,2), axis=1), axis=0)
        vectors = frames[:,edges[:,1]]-frames[:,edges[:,0]]
        length2 = np.square(vectors).sum(-1)
        parallel = vectors@direction
        # |edge + (offset_b-offset_a)*direction| <= ratio*|edge|.
        # Intersect the exact scalar intervals across every supplied pose.
        radius = np.sqrt(np.maximum((maximum_edge_ratio**2-1)*length2+parallel**2, 0))
        lower = ((-parallel-radius)*1000).max(0)
        upper = ((-parallel+radius)*1000).min(0)
        edge_matrix = coo_matrix((np.tile([-1.,1.],len(edges)),
                                 (np.repeat(np.arange(len(edges)),2),edges.ravel())),
                                shape=(len(edges),matrix.shape[1])).tocsr()
        matrix = vstack((matrix, edge_matrix, -edge_matrix), format='csr')
        bounds = np.r_[bounds, lower, -upper]
    inequality_count = matrix.shape[0]
    anchors = None
    attachment_count = 0
    if fixed_attachments is not None:
        ids, weights = map(np.asarray, fixed_attachments)
        if (ids.ndim != 2 or ids.shape[1] != 3 or weights.shape != ids.shape
                or ids.dtype.kind not in 'iu' or not len(ids) or ids.min() < 0
                or ids.max() >= matrix.shape[1] or not np.isfinite(weights).all()
                or (weights < 0).any() or not np.allclose(weights.sum(1), 1)):
            raise ValueError('invalid fixed barycentric attachments')
        anchors = coo_matrix((weights.ravel(), (np.repeat(np.arange(len(ids)), 3), ids.ravel())),
                             shape=(len(ids), matrix.shape[1])).tocsr()
        attachment_count = len(ids)
        # Project the entire attachment subspace at once. Individual row
        # projections converge very slowly for nearby/dependent attachments.
        gram_inverse = np.linalg.pinv((anchors@anchors.T).toarray(), rcond=1e-12, hermitian=True)
    desired = np.asarray(desired_mm, dtype=float)
    if desired.shape != (matrix.shape[1],) or not np.isfinite(desired).all():
        raise ValueError('invalid desired offsets')
    offsets = desired.copy()
    multipliers = np.zeros(matrix.shape[0])
    box_residual = np.zeros_like(offsets)
    anchor_residual = np.zeros_like(offsets)
    active = set()
    converged = False
    for sweep in range(max_sweeps):
        before = offsets.copy()
        active.update(np.flatnonzero((matrix@offsets)[:inequality_count] < bounds-tolerance).tolist())
        for row in sorted(active):
            start, end = matrix.indptr[row:row+2]
            ids, values = matrix.indices[start:end], matrix.data[start:end]
            norm2 = float(values@values)
            if norm2 < 1e-30:
                continue
            # Undo this halfspace's previous projection before reprojecting.
            point = offsets[ids]-multipliers[row]*values
            multiplier = max(0., (bounds[row]-float(values@point))/norm2)
            offsets[ids] = point+multiplier*values
            multipliers[row] = multiplier
        if anchors is not None:
            point = offsets+anchor_residual
            offsets = point-anchors.T@(gram_inverse@(anchors@point))
            anchor_residual = point-offsets
        point = offsets+box_residual
        offsets = np.clip(point, -limit_mm, limit_mm)
        box_residual = point-offsets
        evaluated = matrix@offsets
        ratios = 1+evaluated[:area_count]
        anchor_error = float(np.max(np.abs(anchors@offsets), initial=0)) if anchors is not None else 0.
        inequality_error = float(np.max(bounds-evaluated[:inequality_count], initial=0))
        if (inequality_error <= tolerance and anchor_error <= tolerance
                and np.max(np.abs(offsets-before)) < tolerance):
            converged = True
            break
    report = dict(converged=converged, sweeps=sweep+1, active_constraints=len(active)+attachment_count,
                  minimum_area_ratio=float(ratios.min()), minimum_required=minimum_ratio,
                  offset_tolerance_mm=tolerance, area_ratio_tolerance=tolerance,
                  max_offset_mm=float(np.abs(offsets).max()), limit_mm=limit_mm,
                  fixed_attachments=attachment_count, max_attachment_offset_mm=anchor_error,
                  max_projection_correction_mm=float(np.abs(offsets-desired).max()))
    if edges is not None:
        changed = vectors+(offsets[edges[:,1]]-offsets[edges[:,0]])[None,:,None]*direction*.001
        valid = length2 > 1e-24
        report.update(maximum_edge_ratio_requested=maximum_edge_ratio,
                      maximum_edge_ratio=float(np.sqrt(np.square(changed).sum(-1)[valid]/length2[valid]).max()),
                      edge_interval_violation_mm=float(np.max(bounds[area_count:]-evaluated[area_count:inequality_count], initial=0)))
    return offsets, report
