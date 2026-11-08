"""Compare approximate sclera base-colour profiles in isolated eye EXRs.

This is a regional diagnostic, not a registered full-image metric. It compares
median linear RGB in two annuli normalized by the projected eyeball silhouette.
Speculars and lighting are excluded by using
UE base-color and Cycles emission-of-base-color captures.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .images import read_exr

RINGS = ((.64, .72), (.78, .86))


def fit_strength(reference, baseline, candidate):
    """Bounded least-squares interpolation of two RGB profiles."""
    reference, baseline, candidate = (np.asarray(v, dtype=np.float64) for v in (reference, baseline, candidate))
    if reference.shape != (2, 3) or baseline.shape != reference.shape or candidate.shape != reference.shape:
        raise ValueError('expected two RGB annuli per profile')
    if not all(np.isfinite(v).all() for v in (reference, baseline, candidate)):
        raise ValueError('profiles must be finite')
    delta = candidate - baseline
    return float(np.clip(np.sum((reference - baseline) * delta) / max(float(np.sum(delta * delta)), 1e-12), 0., 1.))


def silhouette(a, *, threshold=.01):
    """Fit the visible eyeball contour, ignoring image-edge clipping."""
    import cv2
    if a.ndim != 3 or a.shape[-1] != 3 or not np.isfinite(a).all():
        raise ValueError('expected finite RGB eye image')
    valid = a.max(-1) > threshold
    h, w = a.shape[:2]
    contours, _ = cv2.findContours(valid.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError('no eye silhouette')
    points = max(contours, key=cv2.contourArea).reshape(-1, 2)
    # A cropped eye has straight edges at the image boundary; fit only genuine silhouette arcs.
    points = points[(points[:, 0] > 2) & (points[:, 0] < w - 3) &
                    (points[:, 1] > 2) & (points[:, 1] < h - 3)]
    if len(points) < 50:
        raise ValueError('insufficient eye silhouette')
    px, py = points[:, 0] / w, points[:, 1] / h
    design = np.stack((2 * px, 2 * py, np.ones(len(points))), axis=1)
    cx, cy, c = np.linalg.lstsq(design, px * px + py * py, rcond=None)[0]
    radius = np.sqrt(c + cx * cx + cy * cy)
    residual = np.sqrt(np.mean((np.hypot(px - cx, py - cy) - radius) ** 2))
    if not .3 < radius < .8 or residual > .02:
        raise ValueError('eye silhouette is not reliably circular')
    return float(cx), float(cy), float(radius), float(residual)


def profile(path):
    a = read_exr(path)[..., :3]
    cx, cy, radius, _ = silhouette(a)
    valid = a.max(-1) > .01
    h, w = a.shape[:2]
    y, x = np.mgrid[:h, :w]
    r = np.hypot(x / w - cx, y / h - cy) / radius
    values = []
    for lo, hi in RINGS:
        pixels = a[(r >= lo) & (r < hi) & valid]
        if len(pixels) < 100:
            raise ValueError(f'insufficient sclera pixels: {path}, annulus {lo}-{hi}')
        values.append(np.median(pixels, axis=0))
    return np.stack(values)


def audit(reference, cycles_root, ids, side, baseline, candidate):
    rows = []
    for sid in ids:
        ref = profile(Path(reference) / sid / f'eye_{side}_basecolor.exr')
        def load(variant):
            return profile(Path(cycles_root) / sid / f'cycles_{variant}__fast' / f'eye_{side}' / 'beauty.exr')
        before, after = load(baseline), load(candidate)
        rows.append(dict(id=sid, baseline=float(np.mean(np.abs(ref - before))),
                         candidate=float(np.mean(np.abs(ref - after))), reference=ref.tolist(),
                         baseline_rgb=before.tolist(), candidate_rgb=after.tolist(),
                         fitted_strength=fit_strength(ref, before, after)))
    return dict(method=__doc__, side=side, baseline=baseline, candidate=candidate, rings=RINGS,
                mean_baseline=float(np.mean([r['baseline'] for r in rows])),
                mean_candidate=float(np.mean([r['candidate'] for r in rows])),
                improved=sum(r['candidate'] < r['baseline'] for r in rows), rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True)
    p.add_argument('--cycles-root', required=True)
    p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--side', default='left', choices=('left', 'right'))
    p.add_argument('--baseline', default='iso_unified_basecolor')
    p.add_argument('--candidate', default='iso_tissue_base')
    p.add_argument('--out', required=True)
    p.add_argument('--fit-out', help='optional JSON of fitted strengths for diagnostic render variants')
    p.add_argument('--rings', choices=('outer', 'limbus'), default='outer')
    a = p.parse_args()
    global RINGS
    if a.rings == 'limbus':
        RINGS = ((.45, .55), (.56, .64))
    result = audit(a.reference, a.cycles_root, a.ids, a.side, a.baseline, a.candidate)
    path = Path(a.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    if a.fit_out:
        fit = Path(a.fit_out); fit.parent.mkdir(parents=True, exist_ok=True)
        fit.write_text(json.dumps(dict(method='bounded least-squares fit to isolated UE sclera base-color annuli',
                                       side=a.side, strengths={r['id']: r['fitted_strength'] for r in result['rows']}),
                                  indent=2) + '\n')
    print(f"{result['improved']}/{len(a.ids)} improved; linear RGB error {result['mean_baseline']:.4f} -> {result['mean_candidate']:.4f}; {path}")


if __name__ == '__main__':
    main()
