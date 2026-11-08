"""MetaHuman eye settings -> vhuman eye parameters (eye/params), calibrated on the UE reference close-ups.

usage: python -m server.vhuman.metahuman.eye_map calibrate          (all reference-rendered samples -> table + report)
       python -m server.vhuman.metahuman.eye_map params <id>        (print the mapped vhuman eye parameters)

vhuman's iris palette (eye/chart.py) is its own, so MetaHuman iris U/V cannot be copied. Instead each sample's iris colour
is measured from the UE base-colour pass of its eye close-up (samples/<id>/renders/eye_<side>__basecolor.exr, linear):
pupil = the dark disk at the centre, iris radius from the camera distance and the GNM limbus radius, inner ring
.45-.65 R and outer ring .65-.90 R. vhuman primary/secondary chart U/V are the nearest chart colours (log RGB) to the inner
/outer measurements (radial blend: primary inside), and global saturation/tint (head.iris_match.color_controls) fix the
remaining difference of the blended colour. Parameters with the same meaning are copied with range conversion
(pupil dilation from the measured pupil/iris ratio, cornea size relative to the MetaHuman default 0.165).
calibrate writes MODELS_ROOT/eye_colour_calibration_v1.json: per sample MetaHuman U/V/blend, measured colours, mapped
vhuman U/V/controls and the chart's reproduction error (Lab dE) -- the colour table the eye prior lacked.
"""
import argparse
import json
import math

import numpy as np

from . import MODELS_ROOT
from .images import read_exr
from .manifest import LICENCE, code_sha256
from .samples import V1
from ..eye import params as P
from ..eye.chart import chart_color

LIMBUS_RADIUS_M = .006          # eye/params optics.limbus_radius default (GNM)
MH_CORNEA_DEFAULT = .165
PUPIL_LUM = .004
VERSION = 2                     # default mapping: v2 sclera tint from the MetaHuman sclera (fast confirm: eye dE -7.5, 12/12 heads); v1 kept
P_REF = P.P_REF                 # pupil radius / iris radius at dilation 1
U_GRID, V_GRID = np.meshgrid(np.linspace(0, 1, 101), np.linspace(0, 1, 101))
CHART = chart_color(U_GRID, V_GRID)


def _srgb_lab(linear):
    from ..reconstruction.view_acceptance import srgb_to_lab
    from ..reconstruction.color_chart import linear_to_srgb
    return srgb_to_lab(np.rint(linear_to_srgb(np.asarray(linear)) * 255).astype(np.uint8))


def _median_iris_ring(img, lum, rr, a, b, sector=None):
    m = (rr >= a) & (rr < b) & (lum > .005)
    if sector is not None:
        m &= sector
    if not m.any():
        raise ValueError('no valid pixels in iris colour ring')
    return np.median(img[m], 0).tolist(), int(m.sum())


def measure(sid, side, *, sector_only=False, geometric_iris=False, radial_bands=False, chroma_detail=False):
    """Iris colours and pupil ratio from the base-colour close-up of eye_<side>."""
    d = V1 / 'samples' / sid
    cams = json.loads((d / 'cameras.json').read_text())
    cam = cams['cameras'][f'eye_{side}']['vhuman']
    img = read_exr(d / 'renders' / f'eye_{side}__basecolor.exr')[..., :3].astype(np.float64)
    h, w = img.shape[:2]
    dist = float(np.linalg.norm(np.asarray(cams['targets'][f'eye_{side}']) - np.asarray(cam['origin'])))
    r_geo = cam['focal_px'] * LIMBUS_RADIUS_M / dist
    yy, xx = np.mgrid[0:h, 0:w]
    lum = img @ np.array([.2126, .7152, .0722])
    near = (xx - w / 2) ** 2 + (yy - h / 2) ** 2 < (2 * r_geo) ** 2
    dark = near & (lum < PUPIL_LUM)                  # the pupil is black in the base-colour pass; dark irises/skin are > .01
    if dark.sum() < 20:
        raise ValueError(f'{sid} {side}: no pupil found')
    cy, cx = np.median(yy[dark]), np.median(xx[dark])
    r_pupil = float(np.sqrt(dark.sum() / np.pi))
    # iris edge: largest step of the median luminance profile in horizontal sectors (+-30 deg; the lids cut the vertical
    # ones), between 1.3 x pupil and 2.5 x the geometric estimate -- the base colour iris renders larger than the GNM limbus
    rad = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2); ang = np.abs(np.arctan2(yy - cy, xx - cx))
    sector = (ang < np.pi / 6) | (ang > 5 * np.pi / 6)
    radii = np.arange(int(1.3 * r_pupil) + 1, int(2.5 * r_geo))
    prof = np.array([np.median(lum[sector & (np.abs(rad - r) < 1.)]) if (sector & (np.abs(rad - r) < 1.)).any() else np.nan for r in radii])
    step = prof[4:] - prof[:-4]
    r_iris = float(radii[2 + int(np.nanargmax(step))]) if np.isfinite(step).any() else r_geo
    if geometric_iris:
        # Soft or weak limbal edges can make the largest gradient land on
        # eyelid/sclera tissue. Keep the camera-derived estimate as a separately
        # testable option rather than silently changing the legacy fit.
        r_iris = r_geo
    rr = rad / r_iris
    pupil_ratio = r_pupil / r_iris

    def ring(a, b):
        return _median_iris_ring(img, lum, rr, a, b, sector if sector_only else None)
    inner, ni = ring(.45, .65); outer, no = ring(.65, .9); whole, nw = ring(max(pupil_ratio + .05, .3), .9)
    # sclera: horizontal sectors 1.15-1.6 R (inside the eye opening at the canthi side), brightest half (vessels excluded)
    sm = sector & (rr >= 1.15) & (rr < 1.6)
    sl = lum[sm]; sclera = np.median(img[sm][sl >= np.median(sl)], 0).tolist() if sm.sum() > 20 else None
    # tighter sclera (eye_map v2): a horizontal band |dy| < .3 R at 1.1-1.45 R, brightest quarter; the sector above reaches
    # the lids on narrow eyes (dark skin then pulls it down: mh_000300 .10/.04/.03 vs .32/.10/.09 here)
    band = (np.abs(yy - cy) < .3 * r_iris) & (np.abs(xx - cx) > 1.1 * r_iris) & (np.abs(xx - cx) < 1.45 * r_iris)
    sclera_band = np.median(img[band & (lum >= np.percentile(lum[band], 75))], 0).tolist() if band.sum() > 40 else None
    from ..eye.iris import angular_contrast, chromatic_fiber_signal
    contrast = angular_contrast(lum, rr, sector)
    radial_linear = [ring(a, b)[0] for a, b in ((.40, .55), (.55, .70), (.70, .85))] if radial_bands else None
    chroma = chromatic_fiber_signal(img, rr, sector, inner, outer) if chroma_detail else None
    return dict(inner_linear=inner, outer_linear=outer, median_linear=whole, angular_contrast=contrast, radial_linear=radial_linear, chroma_fiber_signal=chroma,
                pupil_ratio=pupil_ratio, iris_radius_px=r_iris, geometric_radius_px=float(r_geo),
                center_px=[float(cx), float(cy)], sclera_linear=sclera, sclera_band_linear=sclera_band, pixels=dict(inner=ni, outer=no, whole=nw, sclera=int(sm.sum())))


def nearest_chart(target):
    """vhuman chart (U, V) whose colour is nearest in log RGB."""
    t = np.log(np.asarray(target) + .01)
    err = ((np.log(CHART + .01) - t) ** 2).sum(-1)
    i = np.unravel_index(int(np.argmin(err)), err.shape)
    return float(U_GRID[i]), float(V_GRID[i])


def pattern_number(value):
    """MetaHuman iris pattern -> its IRISnnn number: '<...IrisPattern.IRIS003: 2>' -> 3; a bare int is the enum index."""
    s = str(value)
    if 'IRIS' in s:
        return int(s.split('IRIS')[1][:3])
    return int(value) + 1


def mh_eye(settings, side):
    """Nested MetaHuman eye settings for one side. Preset-jitter samples store the UE struct per eye; uniform samples store
    one flat dict (gen_batch.apply_eyes; unlisted fields keep the UE defaults, here vhuman's)."""
    if f'eye_{side}' in settings:
        e = settings[f'eye_{side}']
        return dict(e, pattern_number=pattern_number(e['iris']['pattern']))
    f = settings
    return dict(pattern_number=int(f['iris_pattern']),
                iris=dict(rotation=f['iris_rotation'], primary_color_u=f['primary_color_u'], primary_color_v=f['primary_color_v'],
                          secondary_color_u=f['secondary_color_u'], secondary_color_v=f['secondary_color_v'], color_blend=f['color_blend'],
                          color_blend_softness=f['color_blend_softness'], shadow_details=f['shadow_details'], limbal_ring_size=f['limbal_ring_size'],
                          limbal_ring_softness=f['limbal_ring_softness'], blend_method=f['blend_method'], global_saturation=f['global_saturation']),
                pupil=dict(dilation=f['pupil_dilation'], feather=f['pupil_feather']),
                cornea=dict(size=f['cornea_size'], limbus_softness=f['cornea_limbus_softness']),
                sclera=dict(transmission_spread=f['sclera_transmission_spread'], vascularity_intensity=f['sclera_vascularity_intensity'],
                            vascularity_coverage=f['sclera_vascularity_coverage']))


def to_vhuman(mh, measured, version=VERSION):
    """mh: params.json eyes.settings.eye_<side>; measured: measure(). Returns validated vhuman eye params + the record.
    version 2: sclera tint = the measured MetaHuman sclera albedo / vhuman's SCLERA_ALBEDO (custom tint), instead of the
    MetaHuman tint setting (measured on a horizontal band beside the iris, measure()['sclera_band_linear']). MetaHuman
    scleras are darker and pinker in base colour (median linear .39/.27/.25 on the 12 comparison heads vs vhuman
    .74/.70/.66); with v1 the Cycles sclera rendered ~2.6x brighter than UE's (lit eye close-ups, 12 heads)."""
    from ..head import iris_match
    p = P.defaults()
    iris, pupil, cornea, sclera = mh['iris'], mh['pupil'], mh['cornea'], mh['sclera']
    d = P.defaults()

    def col(group, key, src):
        return [float(c) for c in src[key][:3]] if key in src else d[group][key]
    p['iris'].update(pattern=f'pattern_{min(max(mh["pattern_number"], 1), 9)}', rotation=float(iris['rotation']) % 1.,
                     color_blend=float(iris['color_blend']), color_blend_softness=max(float(iris['color_blend_softness']), .001),
                     blend_method='Structural' if 'STRUCT' in str(iris['blend_method']).upper() else 'Radial',
                     shadow_details=float(iris['shadow_details']), limbal_ring_size=float(iris['limbal_ring_size']),
                     limbal_ring_softness=float(np.clip(iris['limbal_ring_softness'], .001, .5)),
                     limbal_ring_color=col('iris', 'limbal_ring_color', iris))
    pu, pv = nearest_chart(measured['inner_linear']); su, sv = nearest_chart(measured['outer_linear'])
    p['iris'].update(primary_color_u=pu, primary_color_v=pv, secondary_color_u=su, secondary_color_v=sv)
    blend = .5 * (chart_color(pu, pv) + chart_color(su, sv))
    p['iris'].update(iris_match.color_controls(np.asarray(measured['median_linear']), blend))
    p['pupil'].update(dilation=float(np.clip(measured['pupil_ratio'] / P_REF, .25, 2.)), feather=float(pupil['feather']))
    p['cornea'].update(size=float(np.clip(.2 * float(cornea['size']) / MH_CORNEA_DEFAULT, .1, .3)),
                       limbus_softness=float(np.clip(cornea['limbus_softness'], .001, .15 if version >= 3 else .1)),
                       limbus_color=col('cornea', 'limbus_color', cornea))
    p['sclera'].update(rotation=float(sclera.get('rotation', 0.)) % 1., use_custom_tint=bool(sclera.get('use_custom_tint', False)),
                       tint=[min(c, 1.) for c in col('sclera', 'tint', sclera)],
                       transmission_spread=float(np.clip(sclera['transmission_spread'], .001, .4)),
                       transmission_color=col('sclera', 'transmission_color', sclera),
                       vascularity_intensity=float(np.clip(sclera['vascularity_intensity'], 0, 2 if version >= 3 else 1)),
                       vascularity_coverage=float(np.clip(sclera['vascularity_coverage'], 0, .7)))
    if version >= 2 and measured.get('sclera_band_linear'):
        from ..eye.sclera import SCLERA_ALBEDO
        p['sclera'].update(use_custom_tint=True,
                           tint=np.clip(np.asarray(measured['sclera_band_linear']) / SCLERA_ALBEDO, .05, 1.).tolist())
    p = P.validate(p)
    reproduced = iris_match.controlled_color(blend, p['iris'])
    de = float(np.linalg.norm(_srgb_lab(reproduced) - _srgb_lab(measured['median_linear'])))
    rec = dict(mh=dict(primary_u=iris['primary_color_u'], primary_v=iris['primary_color_v'], secondary_u=iris['secondary_color_u'],
                       secondary_v=iris['secondary_color_v'], color_blend=iris['color_blend'], pattern=mh['pattern_number'],
                       dilation=pupil['dilation'], cornea_size=cornea['size']),
               measured=measured, vhuman=dict(primary=[pu, pv], secondary=[su, sv], global_saturation=p['iris']['global_saturation'],
                                              global_tint=p['iris']['global_tint'], dilation=p['pupil']['dilation'],
                                              sclera_tint=p['sclera']['tint'], sclera_custom_tint=p['sclera']['use_custom_tint']),
               reproduction_delta_e=de, version=version)
    return p, rec


def params_for(sid, side='left', version=VERSION):
    settings = json.loads((V1 / 'samples' / sid / 'params.json').read_text())['eyes']['settings']
    return to_vhuman(mh_eye(settings, side), measure(sid, side, sector_only=version >= 3, geometric_iris=version >= 4), version)


def calibrate(version=VERSION):
    ids = sorted(json.loads((V1 / 'sampling.json').read_text())['render_ids'])
    rows, failed = [], []
    for sid in ids:
        try:
            for side in ('left', 'right'):
                _, rec = params_for(sid, side, version); rows.append(dict(id=sid, side=side, **rec))
        except Exception as e:      # noqa: BLE001
            failed.append(dict(id=sid, error=str(e)))
    de = np.array([r['reproduction_delta_e'] for r in rows])
    out = MODELS_ROOT / 'eye_colour_calibration_v1.json'
    rep = dict(format='vhuman.metahuman_eye_colour.v1', code_sha256=code_sha256(), licence=LICENCE, default_enabled=False,
               method=__doc__.split('\n\n')[1], n=len(rows), failed=failed, mapping_version=version,
               reproduction_delta_e=dict(mean=float(de.mean()), p95=float(np.percentile(de, 95)), max=float(de.max())) if len(de) else None,
               rows=rows)
    out.write_text(json.dumps(rep, indent=1) + '\n')
    return out, rep


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('step', choices=('calibrate', 'params')); ap.add_argument('sid', nargs='?')
    ap.add_argument('--version', type=int, default=VERSION, help='eye mapping version (1: MetaHuman sclera tint setting, 2: measured sclera)')
    a = ap.parse_args()
    if a.step == 'params':
        p, rec = params_for(a.sid, version=a.version)
        print(json.dumps(dict(params=p, record=rec), indent=1, default=float)); return
    out, rep = calibrate(a.version)
    print(out, json.dumps({k: rep[k] for k in ('n', 'reproduction_delta_e')}), 'failed', len(rep['failed']))


if __name__ == '__main__':
    main()
