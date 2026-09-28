"""Conservative geometry-based estimation of broad baked illumination.

A robust log-luminance fit separates a low-order normal-dependent factor
from colour/detail. Single-view albedo and lighting remain ambiguous: use
skin-only samples, require held-out improvement, bound the correction, and
do not extrapolate it onto unseen head surfaces in the caller.
"""
from __future__ import annotations

import numpy as np

LUMA = np.array([.2126, .7152, .0722])


def estimate(normals, rgb, weights):
    """Fit log luminance = intercept + normal dot direction, robustly.

    RGB must be linear. Weights express visible skin support and surface
    area. The returned status is explicit; an unsupported fit is a no-op.
    """
    n = np.asarray(normals, dtype=float)
    c = np.asarray(rgb, dtype=float)
    w = np.asarray(weights, dtype=float)
    if n.ndim != 2 or n.shape[1] != 3 or c.shape != n.shape or w.shape != (len(n),):
        raise ValueError("expected normals/RGB N x 3 and weights N")
    if not np.isfinite(n).all() or not np.isfinite(c).all() or not np.isfinite(w).all() or (w < 0).any() or (c < 0).any():
        raise ValueError("samples must be finite with nonnegative RGB and weights")
    norm = np.linalg.norm(n, axis=1)
    lum = c @ LUMA
    valid = (w > 0) & (norm > .5) & (lum > .003) & (lum < .95)
    # Keep validation membership stable if a marginal sample enters/leaves
    # the skin gate. Assigning after filtering would shift every later row.
    fold = np.flatnonzero(valid) % 5
    n, w, lum = n[valid] / norm[valid, None], w[valid], lum[valid]
    result = {"status": "insufficient support", "samples": len(n),
              "direction": [0., 0., 0.], "center": [0., 0., 0.]}
    if len(n) < 64:
        return result
    if min(np.bincount(fold, minlength=5)) < 12:
        result["status"] = "insufficient validation support"
        return result
    w = w / w.mean()
    center = np.average(n, axis=0, weights=w)
    x = n - center
    covariance = (x.T * w) @ x / w.sum()
    if np.linalg.eigvalsh(covariance).min() < .001:
        result["status"] = "insufficient normal variation"
        return result
    design = np.column_stack([np.ones(len(n)), x])
    target = np.log(lum)

    def solve(select):
        a, y, base = design[select], target[select], w[select]
        robust = np.ones(len(y))
        # Small ridge discourages large coefficients on poorly sampled axes.
        penalty = np.diag([0., .001, .001, .001]) * base.sum()
        coefficient = np.zeros(4)
        for _ in range(12):
            effective = base * robust
            coefficient = np.linalg.solve((a.T * effective) @ a + penalty,
                                          a.T @ (effective * y))
            residual = y - a @ coefficient
            scale = max(float(np.median(np.abs(residual - np.median(residual)))) * 1.4826, .015)
            robust = np.minimum(1., 1.5 * scale / np.maximum(np.abs(residual), 1e-12))
        return coefficient

    # Remove global level when comparing errors: the goal is to reduce
    # direction-dependent illumination, not merely change exposure.
    def spread(values, weight):
        mean = np.average(values, weights=weight)
        return float(np.sqrt(np.average((values-mean)**2, weights=weight)))
    residual = np.empty(len(n))
    improving = 0
    folds = []
    for k in range(5):
        holdout = fold == k
        coefficient = solve(~holdout)
        residual[holdout] = target[holdout] - design[holdout] @ coefficient
        before_fold = spread(target[holdout], w[holdout])
        after_fold = spread(residual[holdout], w[holdout])
        improving += after_fold < before_fold
        folds.append([before_fold, after_fold])
    before = spread(target, w)
    after = spread(residual, w)
    result.update(heldout_log_std_before=before, heldout_log_std_after=after,
                  improving_folds=int(improving), validation_folds=folds)
    if before < .035 or after > .95 * before or improving < 4:
        result["status"] = "no reliable directional component"
        return result
    coefficient = solve(np.ones(len(n), bool))
    direction = coefficient[1:]
    magnitude = float(np.linalg.norm(direction))
    direction *= min(1., 1.2 / max(magnitude, 1e-12))
    result.update(status="estimated", direction=direction.tolist(), center=center.tolist(),
                  correction_limit_stops=1.)
    return result


def gain(normals, estimate, strength=1.):
    """Achromatic scalar correction, bounded to one stop at full strength."""
    n = np.asarray(normals, dtype=float)
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    if estimate["status"] != "estimated" or strength == 0:
        return np.ones(n.shape[:-1])
    n = n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-12)
    log_shading = (n - np.asarray(estimate["center"])) @ np.asarray(estimate["direction"])
    return np.exp(-strength * np.clip(log_shading, -np.log(2.), np.log(2.)))
