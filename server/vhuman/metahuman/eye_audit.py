"""Eye audits of the UE-vs-Cycles eye close-ups (CPU, image-only; 2026-11-05).

placement: pupil centre / iris radius / eye-opening box in the UE and Cycles halves of compare_v1/<id>/
           eye_<side>_ue_vhstudio_<variant>_cycles.png (512 px each). Detection fails on ~1/4 of eyes (dark irises, lids),
           so summaries use only eyes whose centre lands near the frame centre in both halves.
color:     HSV of the vhuman iris texture annulus (cycles_<variant>/eye_<side>/iris.png) vs the rendered iris ring in
           both halves -- separates the colour mapping (texture) from the shading (render).
usage: python -m server.vhuman.metahuman.eye_audit [--variant normal_eyes2__fast] [--out audit.json]
"""
import argparse
import colorsys
import json

import numpy as np
from PIL import Image

from .compare_render import COMPARE_ROOT


def measure(img):
    from scipy import ndimage
    h, w = img.shape[:2]
    lum = img.astype(float).mean(-1); sat = img.max(-1).astype(float) - img.min(-1)
    yy, xx = np.mgrid[0:h, 0:w]
    centre = (np.abs(xx - w / 2) < .3 * w) & (np.abs(yy - h / 2) < .2 * h)
    lab, n = ndimage.label(centre & (lum <= max(np.percentile(lum[centre], 1.5), 25)))     # pupil: darkest blob
    if not n:
        return None
    k = 1 + int(np.argmax(ndimage.sum(np.ones_like(lum), lab, range(1, n + 1))))
    cy, cx = ndimage.center_of_mass(lab == k)
    scl = centre & (lum > np.percentile(lum[centre], 80)) & (sat < np.percentile(sat[centre], 30))
    rad = np.hypot(xx - cx, yy - cy); ang = np.abs(np.arctan2(yy - cy, xx - cx))
    sector = (ang < np.pi / 6) | (ang > 5 * np.pi / 6)
    radii = np.arange(8, int(.2 * w))
    prof = np.array([np.median(lum[sector & (np.abs(rad - r) < 1)]) for r in radii])
    r_iris = float(radii[2 + int(np.argmax(prof[5:] - prof[:-5]))])
    lab, _ = ndimage.label(scl | (rad < r_iris)); m = lab == lab[int(cy), int(cx)]
    ys, xs = np.nonzero(m)
    return dict(cx=cx / w, cy=cy / h, r_iris=r_iris / w, open_x0=xs.min() / w, open_x1=xs.max() / w,
                open_h=(ys.max() - ys.min()) / h, open_cy=(ys.max() + ys.min()) / 2 / h)


def _hsv(rgb):
    h, s, v = colorsys.rgb_to_hsv(*(np.asarray(rgb) / 255.)); return [round(h * 360), round(s, 3), round(v, 3)]


def _ring(img, cx, cy, r0, r1):
    h, w = img.shape[:2]; yy, xx = np.mgrid[0:h, 0:w]; rr = np.hypot(xx - cx, yy - cy)
    m = (rr > r0) & (rr < r1); lum = img.mean(-1); m &= lum < np.percentile(lum[m], 80)        # drop highlights
    return np.median(img[m], 0)


def audit(variant='normal_eyes2__fast'):
    rows = []
    for d in sorted(COMPARE_ROOT.glob('mh_*')):
        for side in ('left', 'right'):
            p = d / f'eye_{side}_ue_vhstudio_{variant}_cycles.png'
            if not p.is_file():
                continue
            a = np.asarray(Image.open(p).convert('RGB')).astype(float); w = a.shape[1] // 2
            u, c = measure(a[:, :w]), measure(a[:, w:])
            if not (u and c):
                continue
            row = dict(sid=d.name, side=side, ue=u, cycles=c, good=all(.42 < x['cx'] < .56 and .43 < x['cy'] < .52 for x in (u, c)))
            tex = d / f'cycles_{variant}' / f'eye_{side}' / 'iris.png'
            if row['good'] and tex.is_file():
                t = np.asarray(Image.open(tex).convert('RGB')).astype(float)
                row['hsv'] = dict(texture=_hsv(_ring(t, t.shape[1] / 2, t.shape[0] / 2, .18 * t.shape[0], .39 * t.shape[0])),
                                  ue=_hsv(_ring(a[:, :w], u['cx'] * w, u['cy'] * w, 9, 22)),
                                  cycles=_hsv(_ring(a[:, w:], c['cx'] * w, c['cy'] * w, 9, 22)))
            rows.append(row)
    good = [r for r in rows if r['good']]
    dx = np.array([r['cycles']['cx'] - r['ue']['cx'] for r in good]) * 512; dy = np.array([r['cycles']['cy'] - r['ue']['cy'] for r in good]) * 512
    sat = np.array([[r['hsv'][k][1] for k in ('texture', 'ue', 'cycles')] for r in good if 'hsv' in r])
    summary = dict(variant=variant, eyes=len(rows), good=len(good), pupil_dx_px_median=float(np.median(dx)) if len(dx) else None,
                   pupil_dy_px_median=float(np.median(dy)) if len(dy) else None,
                   iris_sat_median_texture_ue_cycles=np.median(sat, 0).round(3).tolist() if len(sat) else None)
    return dict(summary=summary, rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--variant', default='normal_eyes2__fast'); ap.add_argument('--out')
    a = ap.parse_args()
    out = audit(a.variant)
    print(json.dumps(out['summary']))
    if a.out:
        open(a.out, 'w').write(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
