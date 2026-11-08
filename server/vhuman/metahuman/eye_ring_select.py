"""Select an experimental limbal-ring treatment from an unzoomed UE reference.

The comparison uses reference-derived per-eye color fits and approximate radial
registration, so it is not a reference-free predictor or a perceptual score.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from ..eye.optics import srgb_to_linear
from .eye_map import measure
from .eye_ring_audit import LUMA, ring_detail
from .images import read_exr
from .samples import V1

RADII = np.arange(.4, .901, .01)


def select_errors(baseline, neutral):
    """Choose neutral only when both finite errors exist and it is lower."""
    return 'neutral' if np.isfinite(baseline) and np.isfinite(neutral) and neutral < baseline else 'baseline'


def select(sid, base, side='left'):
    base = Path(base)
    if side not in ('left', 'right'):
        raise ValueError('eye side must be left or right')
    params = json.loads((base / ('eye_params_right.json' if side == 'right' else 'eye_params.json')).read_text())
    measured = measure(sid, side, sector_only=True, geometric_iris=True)
    a = read_exr(V1 / 'samples' / sid / 'renders' / f'eye_{side}__basecolor.exr')[..., :3]
    y, x = np.mgrid[:a.shape[0], :a.shape[1]]
    cx, cy = measured['center_px']
    radius = np.hypot(x - cx, y - cy) / measured['iris_radius_px']
    sector = np.abs(y - cy) < .5 * np.abs(x - cx)

    def radial(lum, r, mask):
        values = []
        for target in RADII:
            selected = mask & (np.abs(r - target) < .012)
            if selected.sum() < 8:
                raise ValueError(f'{sid}: insufficient samples near iris radius {target:.2f}')
            values.append(float(np.median(lum[selected])))
        return np.asarray(values)

    valid = (RADII > measured['pupil_ratio'] + .05) & (RADII < .86)
    if valid.sum() < 8:
        raise ValueError(f'{sid}: pupil leaves too few iris bins')

    def detail(values):
        return ring_detail(values / max(float(np.median(values[valid])), .005))

    reference = detail(radial(a @ LUMA, radius, sector))
    errors = []
    names = (('eye_unified_right_bands_select.png', 'eye_unified_right_ring_none_select.png') if side == 'right' else
             ('eye_unified_clean_bands_trans05.png', 'eye_unified_clean_ring_none.png'))
    for name in names:
        c = srgb_to_linear(np.asarray(Image.open(base / name).convert('RGB'), np.float64) / 255.)
        h, w = c.shape[:2]
        if h != w:
            raise ValueError(f'{base / name}: expected a square texture')
        ty, tx = np.mgrid[:h, :w]
        tr = np.hypot((tx + .5) / w - .5, (ty + .5) / h - .5) / params['cornea']['size']
        result = detail(radial(c @ LUMA, tr, np.ones_like(tr, bool)))
        errors.append(float(np.mean(np.abs(result[valid] - reference[valid]))))
    return dict(baseline_error=errors[0], neutral_error=errors[1], choice=select_errors(*errors), bins=int(valid.sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id-base', nargs='+', required=True, help='sample_id=staged_base_directory')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    samples = {}
    for item in args.id_base:
        sid, separator, base = item.partition('=')
        if not separator or not sid or not base or sid in samples:
            parser.error(f'invalid or duplicate id-base: {item}')
        samples[sid] = select(sid, base)
    result = dict(method=__doc__, samples=samples)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')
    print(f'{sum(v["choice"] == "neutral" for v in samples.values())}/{len(samples)} select neutral; {out}')


if __name__ == '__main__':
    main()
