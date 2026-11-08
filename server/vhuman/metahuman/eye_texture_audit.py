"""Compare medium-frequency sclera contrast in isolated base-colour EXRs.

The statistic is a standard deviation of difference-of-Gaussian luminance in a
silhouette-normalized annulus, not a perceptual score or a vessel detector.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .eye_tissue_audit import silhouette
from .images import read_exr


def contrast(path):
    import cv2
    a = read_exr(path)[..., :3]
    if not np.isfinite(a).all():
        raise ValueError(f'nonfinite eye EXR: {path}')
    a = cv2.resize(a, (512, 512), interpolation=cv2.INTER_AREA)
    cx, cy, radius, _ = silhouette(a)
    y, x = np.mgrid[:512, :512]
    r = np.hypot(x / 512 - cx, y / 512 - cy) / radius
    lum = a @ np.array([.2126, .7152, .0722])
    detail = cv2.GaussianBlur(lum, (0, 0), 1.5) - cv2.GaussianBlur(lum, (0, 0), 8.)
    region = (r > .55) & (r < .8) & (a.max(-1) > .01)
    if region.sum() < 100:
        raise ValueError(f'insufficient sclera pixels: {path}')
    return float(np.std(detail[region]))


def audit(reference, cycles_root, ids, side, baseline, candidate):
    rows = []
    for sid in ids:
        ref = contrast(Path(reference) / sid / f'eye_{side}_basecolor.exr')
        def load(variant):
            return contrast(Path(cycles_root) / sid / f'cycles_{variant}__fast' / f'eye_{side}' / 'beauty.exr')
        before, after = load(baseline), load(candidate)
        rows.append(dict(id=sid, reference=ref, baseline=before, candidate=after,
                         error_before=abs(ref - before), error_after=abs(ref - after)))
    return dict(method=__doc__, side=side, baseline=baseline, candidate=candidate,
                improved=sum(r['error_after'] < r['error_before'] for r in rows),
                mean_error_before=float(np.mean([r['error_before'] for r in rows])),
                mean_error_after=float(np.mean([r['error_after'] for r in rows])), rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True); p.add_argument('--cycles-root', required=True)
    p.add_argument('--ids', nargs='+', required=True); p.add_argument('--side', choices=('left', 'right'), default='left')
    p.add_argument('--baseline', required=True); p.add_argument('--candidate', required=True); p.add_argument('--out', required=True)
    a = p.parse_args()
    result = audit(a.reference, a.cycles_root, a.ids, a.side, a.baseline, a.candidate)
    path = Path(a.out); path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(result, indent=2) + '\n')
    print(f"{result['improved']}/{len(a.ids)} improved; detail contrast error {result['mean_error_before']:.5f} -> {result['mean_error_after']:.5f}; {path}")


if __name__ == '__main__':
    main()
