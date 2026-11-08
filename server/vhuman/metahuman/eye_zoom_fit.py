"""Fit diagnostic eye-camera zooms from UE and Cycles eyeball silhouettes.

The fit uses per-eye reference images and is for controlled comparison only;
it does not estimate a camera from a portrait or alter the default camera.
"""
import argparse
import json
from pathlib import Path

from .eye_tissue_audit import silhouette
from .images import read_exr


def fit(reference, cycles_root, ids, variant='iso_tissue1', source_zoom=2.5):
    rows = {}
    for sid in ids:
        rows[sid] = {}
        for side in ('left', 'right'):
            ref = read_exr(Path(reference) / sid / f'eye_{side}.exr')[..., :3]
            cy = read_exr(Path(cycles_root) / sid / f'cycles_{variant}__fast' / f'eye_{side}' / 'beauty.exr')[..., :3]
            ucx, ucy, ur, ue = silhouette(ref, threshold=.001)
            ccx, ccy, cr, ce = silhouette(cy, threshold=.001)
            zoom = source_zoom * ur / cr
            if not 1. <= zoom <= 4.:
                raise ValueError(f'unreasonable zoom fit for {sid} {side}: {zoom}')
            rows[sid][side] = dict(zoom=zoom, ue_center=[ucx, ucy], cycles_center=[ccx, ccy],
                                    ue_radius=ur, cycles_radius=cr, ue_residual=ue, cycles_residual=ce)
    return dict(method='match fitted isolated eyeball silhouette radii by focal length', source_zoom=source_zoom,
                baseline=variant, samples=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True); p.add_argument('--cycles-root', required=True)
    p.add_argument('--ids', nargs='+', required=True); p.add_argument('--variant', default='iso_tissue1')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    result = fit(a.reference, a.cycles_root, a.ids, a.variant)
    path = Path(a.out); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n')
    print(f"fitted {2 * len(a.ids)} eye-camera zooms; {path}")


if __name__ == '__main__':
    main()
