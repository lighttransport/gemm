"""Limit a sparse mesh corrective with shared vertex weights across a sequence.

The saved poses and their linear midpoints are checked. This is an orientation
bound, not a continuous collision or anatomical acceptance guarantee.
"""
import numpy as np


def constrain_sequence(original, desired, triangles, *, minimum_ratio=.2, max_iterations=40):
    original, desired = np.asarray(original), np.asarray(desired)
    triangles = np.asarray(triangles)
    if (original.ndim != 3 or original.shape[-1] != 3 or not len(original)
            or original.shape != desired.shape or not np.isfinite(original).all()
            or not np.isfinite(desired).all() or triangles.ndim != 2 or triangles.shape[1] != 3
            or triangles.dtype.kind not in 'iu' or not len(triangles)
            or triangles.min() < 0 or triangles.max() >= original.shape[1]
            or not 0 < minimum_ratio < 1 or max_iterations < 1):
        raise ValueError('invalid sequence corrective')
    base = original.astype(np.float64)
    delta = desired.astype(np.float64) - base
    moving = np.any(delta != 0, axis=(0, 2))
    faces = triangles[moving[triangles].any(1)]
    weight = np.ones(original.shape[1])
    history = []
    if not len(faces):
        return original.astype(np.float32).copy(), weight, history
    for iteration in range(max_iterations):
        candidate = (base + delta * weight[None, :, None]).astype(np.float32)
        bad = set()
        minimum = 1.
        for sample in range(2 * len(base) - 1):
            left = sample // 2
            right = min(left + 1, len(base) - 1)
            alpha = (sample % 2) * .5
            p = base[left, faces] * (1 - alpha) + base[right, faces] * alpha
            q = (candidate[left, faces].astype(float) * (1 - alpha)
                 + candidate[right, faces].astype(float) * alpha)
            normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            den = (normal * normal).sum(1)
            valid = den > 1e-24
            ratio = ((np.cross(q[:, 1] - q[:, 0], q[:, 2] - q[:, 0]) * normal).sum(1)
                     / np.maximum(den, 1e-30))
            if valid.any():
                minimum = min(minimum, float(ratio[valid].min()))
            bad.update(faces[valid & (ratio < minimum_ratio)].ravel())
        history.append(dict(iteration=iteration, minimum_relative_area=minimum,
                            constrained_vertices=len(bad)))
        if not bad:
            return candidate, weight, history
        ids = np.array(sorted(bad), int)
        weight[ids] *= .5
    raise ValueError('sequence corrective did not meet the orientation bound')
