"""Compare narrow radial iris structure in isolated base-color renders.

Profiles use horizontal sectors and normalize by each eye's median iris
luminance. This is a diagnostic, not a perceptual metric: pupil localization,
imperfect camera registration, highlights and differences in iris size can
all affect the result. The reference also informed the texture fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .eye_tissue_audit import silhouette
from .images import read_exr

RADII = np.arange(.16, .401, .005)
LUMA = np.array([.2126, .7152, .0722])


def profile(path):
    a = read_exr(path)[..., :3].astype(np.float64)
    if a.ndim != 3 or not np.isfinite(a).all():
        raise ValueError(f'invalid eye EXR: {path}')
    h, w = a.shape[:2]; y, x = np.mgrid[:h, :w]; lum = a @ LUMA
    dark = (((x / w - .5) ** 2 + (y / h - .5) ** 2) < .16 ** 2) & (lum < .004)
    if dark.sum() < 20:
        raise ValueError(f'no pupil found: {path}')
    cx, cy = np.median(x[dark]), np.median(y[dark])
    _, _, sphere, _ = silhouette(a)
    pupil = float(np.sqrt(dark.sum() / np.pi) / w / sphere)
    radius = np.hypot((x - cx) / w, (y - cy) / h) / sphere
    sector = np.abs(y - cy) < .5 * np.abs(x - cx)
    values = []
    for r in RADII:
        mask = sector & (np.abs(radius - r) < .004)
        if mask.sum() < 20:
            raise ValueError(f'insufficient radial samples: {path}')
        values.append(np.median(lum[mask]))
    return np.asarray(values), pupil


def ring_detail(values):
    """Remove a broad radial trend; retain narrow changes in the profile."""
    x = np.arange(-8, 9, dtype=np.float64)
    kernel = np.exp(-.5 * (x / 3.) ** 2); kernel /= kernel.sum()
    smooth = np.convolve(np.pad(values, 8, mode='edge'), kernel, mode='valid')
    return values - smooth


def audit(reference, cycles_root, ids, side, baseline, candidate):
    rows = []
    for sid in ids:
        ref, rp = profile(Path(reference) / sid / f'eye_{side}_basecolor.exr')
        def load(variant):
            return profile(Path(cycles_root) / sid / f'cycles_{variant}__fast' / f'eye_{side}' / 'beauty.exr')
        before, bp = load(baseline); after, ap = load(candidate)
        valid = (RADII > max(rp, bp, ap) + .03) & (RADII < .37)
        if valid.sum() < 8:
            raise ValueError(f'{sid}: pupil leaves too few iris bins')
        details = [ring_detail(v / max(float(np.median(v[valid])), .005)) for v in (ref, before, after)]
        errors = [float(np.mean(np.abs(d[valid] - details[0][valid]))) for d in details[1:]]
        rows.append(dict(id=sid, before=errors[0], after=errors[1], bins=int(valid.sum())))
    return dict(method=__doc__, baseline=baseline, candidate=candidate, side=side, rows=rows,
                improved=sum(r['after'] < r['before'] for r in rows),
                mean_before=float(np.mean([r['before'] for r in rows])),
                mean_after=float(np.mean([r['after'] for r in rows])))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True); p.add_argument('--cycles-root', required=True)
    p.add_argument('--ids', nargs='+', required=True); p.add_argument('--side', choices=('left', 'right'), default='left')
    p.add_argument('--baseline', required=True); p.add_argument('--candidate', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    result = audit(a.reference, a.cycles_root, a.ids, a.side, a.baseline, a.candidate)
    path = Path(a.out); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(result, indent=2) + '\n')
    print(f"{result['improved']}/{len(a.ids)} improved; radial detail error {result['mean_before']:.5f} -> {result['mean_after']:.5f}; {path}")


if __name__ == '__main__':
    main()
