"""Eye audits of the UE-vs-Cycles eye close-ups (CPU, image-only; 2026-11-05).

placement: pupil centre / iris radius / eye-opening box in the UE and Cycles halves of compare_v1/<id>/
           eye_<side>_ue_vhstudio_<variant>_cycles.png (512 px each). Detection fails on ~1/4 of eyes (dark irises, lids),
           so summaries use only eyes whose centre lands near the frame centre in both halves.
color:     HSV of the vhuman iris texture annulus (cycles_<variant>/eye_<side>/iris.png) vs the rendered iris ring in
           both halves -- separates the colour mapping (texture) from the shading (render).
usage: python -m server.vhuman.metahuman.eye_audit [--variant normal_eyes2__fast] [--out audit.json]
       python -m server.vhuman.metahuman.eye_audit --variant combo__fast --compare c_iris_emit_1__fast
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


def compare_audits(baseline, candidate, min_saturation=.1):
    """Compare only eyes with valid colour samples in both audits; hue is circular and undefined for grey pixels."""
    def valid(a):
        return {(r['sid'], r['side']): r['hsv'] for r in a['rows'] if r['good'] and 'hsv' in r}

    a, b = valid(baseline), valid(candidate)
    keys = sorted(a.keys() & b.keys())
    rows = []
    for sid, side in keys:
        key = (sid, side)
        # Pupil detection may choose different UE sample locations for the two candidates.
        row = dict(sid=sid, side=side)
        for name, hsv in (('baseline', a[key]), ('candidate', b[key])):
            ue, cyc = hsv['ue'], hsv['cycles']
            row[name] = dict(saturation_error=abs(float(cyc[1]) - float(ue[1])),
                             hue_error_degrees=(abs((float(cyc[0]) - float(ue[0]) + 180) % 360 - 180)
                                                if min(cyc[1], ue[1]) >= min_saturation else None))
        rows.append(row)
    summary = dict(baseline_variant=baseline.get('summary', {}).get('variant'),
                   candidate_variant=candidate.get('summary', {}).get('variant'),
                   matched_eyes=len(rows), hue_min_saturation=min_saturation)
    for name in ('baseline', 'candidate'):
        sat = [r[name]['saturation_error'] for r in rows]
        hue = [r[name]['hue_error_degrees'] for r in rows if r[name]['hue_error_degrees'] is not None]
        summary[name] = dict(median_saturation_error=float(np.median(sat)) if sat else None,
                             hue_eyes=len(hue), median_hue_error_degrees=float(np.median(hue)) if hue else None)
    summary['candidate_saturation_closer'] = sum(r['candidate']['saturation_error'] < r['baseline']['saturation_error'] for r in rows)
    paired_hue = [r for r in rows if all(r[n]['hue_error_degrees'] is not None for n in ('baseline', 'candidate'))]
    summary['paired_hue_eyes'] = len(paired_hue)
    summary['candidate_hue_closer'] = sum(r['candidate']['hue_error_degrees'] < r['baseline']['hue_error_degrees'] for r in paired_hue)
    summary['paired_median_hue_error_degrees'] = {
        name: float(np.median([r[name]['hue_error_degrees'] for r in paired_hue])) if paired_hue else None
        for name in ('baseline', 'candidate')}
    return dict(summary=summary, rows=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--variant', default='normal_eyes2__fast'); ap.add_argument('--out')
    ap.add_argument('--compare', help='compare this candidate with --variant on the same valid eyes')
    a = ap.parse_args()
    out = audit(a.variant)
    if a.compare:
        out = compare_audits(out, audit(a.compare))
    print(json.dumps(out['summary']))
    if a.out:
        open(a.out, 'w').write(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
