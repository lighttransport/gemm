"""Audit approximate inner-iris colour in isolated base-colour EXRs.

Find the dark pupil separately in each render and compare median linear RGB in
horizontal sectors of a silhouette-normalized annulus. This diagnostic is not an
image registration or an assessment of iris shape, texture, reflections or
complete perceptual quality. MetaHuman colour fitting already uses references.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .images import read_exr
from .eye_tissue_audit import silhouette

RING = (.22, .30)


def profile(path):
    a = read_exr(path)[..., :3]
    if a.ndim != 3 or a.shape[-1] != 3 or not np.isfinite(a).all():
        raise ValueError(f'invalid eye EXR: {path}')
    h, w = a.shape[:2]; y, x = np.mgrid[:h, :w]
    lum = a @ np.array([.2126, .7152, .0722])
    dark = (((x / w - .5) ** 2 + (y / h - .5) ** 2) < .16 ** 2) & (lum < .004)
    if dark.sum() < 20:
        raise ValueError(f'no pupil found: {path}')
    cx, cy = np.median(x[dark]), np.median(y[dark])
    pupil = np.sqrt(dark.sum() / np.pi) / w
    _, _, sphere_radius, _ = silhouette(a)
    r = np.hypot((x - cx) / w, (y - cy) / h) / sphere_radius
    region = (r > RING[0]) & (r < RING[1]) & (np.abs(y - cy) < .5 * np.abs(x - cx))
    if pupil / sphere_radius >= RING[0] - .02 or region.sum() < 100:
        raise ValueError(f'iris annulus overlaps pupil or is empty: {path}')
    return dict(rgb=np.median(a[region], axis=0).tolist(), center=[float(cx / w), float(cy / h)],
                pupil_radius=float(pupil), sphere_radius=sphere_radius, pixels=int(region.sum()))


def audit(reference, cycles_root, ids, side, baseline, candidate):
    rows = []
    for sid in ids:
        ref = profile(Path(reference) / sid / f'eye_{side}_basecolor.exr')
        def load(variant):
            return profile(Path(cycles_root) / sid / f'cycles_{variant}__fast' / f'eye_{side}' / 'beauty.exr')
        before, after = load(baseline), load(candidate)
        err = lambda p: float(np.mean(np.abs(np.asarray(ref['rgb']) - p['rgb'])))
        rows.append(dict(id=sid, baseline=err(before), candidate=err(after), reference=ref,
                         baseline_profile=before, candidate_profile=after))
    return dict(method=__doc__, side=side, annulus=RING, baseline=baseline, candidate=candidate,
                mean_baseline=float(np.mean([r['baseline'] for r in rows])),
                mean_candidate=float(np.mean([r['candidate'] for r in rows])),
                improved=sum(r['candidate'] < r['baseline'] for r in rows), rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True); p.add_argument('--cycles-root', required=True)
    p.add_argument('--ids', nargs='+', required=True); p.add_argument('--side', choices=('left', 'right'), default='left')
    p.add_argument('--baseline', default='iso_unified_basecolor'); p.add_argument('--candidate', required=True)
    p.add_argument('--ring', choices=('inner', 'outer'), default='inner')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    global RING
    if a.ring == 'outer':
        RING = (.30, .40)
    result = audit(a.reference, a.cycles_root, a.ids, a.side, a.baseline, a.candidate)
    path = Path(a.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(f"{result['improved']}/{len(a.ids)} improved; linear RGB error {result['mean_baseline']:.4f} -> {result['mean_candidate']:.4f}; {path}")


if __name__ == '__main__':
    main()
