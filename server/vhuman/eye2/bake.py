"""Build layered-eye assets for one eye: meshes (npz, eye-local frame) + iris/sclera textures (PNG).

Identity inputs (labelled UE-derived): the per-eye vhuman parameter dict (``eye_params*.json``: optics, structure,
vascularity) and colour medians measured from that eye's staged angular base-colour texture (iris inner/outer zone,
sclera near/far, pupil/iris radius). Nothing here is fitted to rendered UE images.

    python -m server.vhuman.eye2.bake <base dir> [--side left|right|both] [--fast]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from . import geometry as G
from . import textures as T

ASSET_VERSION = 20
SOURCE = dict(left=('eye_unified_clean_ring_none.png', 'eye_unified.png'),
              right=('eye_unified_right_bilateral.png', 'eye_unified_right.png', 'eye_unified.png'))
PARAMS = dict(left=('eye_params.json',), right=('eye_params_right.json', 'eye_params.json'))


def srgb_to_linear(x):
    x = np.asarray(x, np.float64)
    return np.where(x <= .04045, x / 12.92, ((x + .055) / 1.055) ** 2.4)


def linear_to_srgb_u8(x):
    x = np.clip(np.asarray(x, np.float64), 0, 1)
    s = np.where(x <= .0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - .055)
    return np.clip(np.rint(s * 255), 0, 255).astype(np.uint8)


def u16(x):
    return Image.fromarray(np.clip(np.rint(np.asarray(x) * 65535), 0, 65535).astype(np.uint16))


def _first(base: Path, names):
    for n in names:
        if (base / n).is_file():
            return base / n
    raise FileNotFoundError(f'none of {names} in {base}')


def measure(img: np.ndarray) -> dict:
    """Radial medians of an angular eye base-colour texture (centre = optical axis, UV radius .2 ~ cornea edge).
    Only the horizontal sectors are used (eyelid-free in the source captures)."""
    lin = srgb_to_linear(img[..., :3] / 255.)
    h, w = lin.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    u, v = (x + .5) / w - .5, (y + .5) / h - .5
    r = np.hypot(u, v); ang = np.abs(np.arctan2(v, u))
    horiz = (ang < np.radians(35)) | (ang > np.radians(145))
    lum = lin @ np.array([.2126, .7152, .0722])
    edges = np.linspace(.01, .3, 117); mids = .5 * (edges[1:] + edges[:-1])
    prof = np.array([np.median(lum[horiz & (r >= a) & (r < b)]) for a, b in zip(edges[:-1], edges[1:])])
    # pupil edge: first rise above the midpoint of pupil min and iris median; iris edge: first rise toward sclera
    p_min = prof[:6].min(); iris_med = np.median(prof[(mids > .07) & (mids < .14)])
    r_pupil = float(mids[np.argmax(prof > .5 * (p_min + iris_med))])
    scl = np.median(prof[(mids > .24) & (mids < .3)])
    r_iris = float(mids[np.argmax((prof > .5 * (iris_med + scl)) & (mids > r_pupil + .03))])
    def med(lo, hi):
        sel = horiz & (r >= lo) & (r < hi)
        return np.median(lin[sel], 0)
    floor = 1e-3
    inner = med(r_pupil + .1 * (r_iris - r_pupil), r_pupil + .3 * (r_iris - r_pupil))
    outer = med(r_pupil + .45 * (r_iris - r_pupil), r_pupil + .8 * (r_iris - r_pupil))
    near = med(r_iris + .03, r_iris + .08)
    far = np.median(lin[(r > .42) & (r < .5)], 0)
    inner, outer = np.maximum(inner, floor), np.maximum(outer, floor)
    rim = np.maximum(med(r_pupil + .85 * (r_iris - r_pupil), r_pupil + .97 * (r_iris - r_pupil)), floor)
    return dict(r_pupil=r_pupil, r_iris=r_iris, pupil_frac=r_pupil / r_iris, inner=inner.tolist(),
                outer=outer.tolist(), rim=rim.tolist(), near=near.tolist(), far=far.tolist())


def measure_basecolor_exr(path, pupil_frac: float, r_iris: float = .4875) -> dict:
    """Same statistics from a UE base-colour capture of the isolated eyeball (screen space, front view): radii are
    fractions of the fitted silhouette radius (r = sin(theta) for the sphere). These are material inputs, not lit pixels."""
    from ..metahuman.images import read_exr
    b = np.asarray(read_exr(str(path))[..., :3], np.float64)
    h, w = b.shape[:2]; y, x = np.mgrid[0:h, 0:w]
    m = b.sum(-1) > 1e-4; cy, cx = y[m].mean(), x[m].mean(); R = math.sqrt(m.sum() / math.pi)
    r = np.hypot(y - cy, x - cx) / R; horiz = m & (np.abs(y - cy) < .35 * R)
    # radii from geometry (anatomical limbus ratio) and the texture-measured pupil: edge detection on base colour is
    # unreliable (the pupil is not black in every capture)
    r_pupil = pupil_frac * r_iris
    def med(lo, hi):
        return np.median(b[horiz & (r >= lo) & (r < hi)], 0)
    inner = np.maximum(med(r_pupil + .1 * (r_iris - r_pupil), r_pupil + .3 * (r_iris - r_pupil)), 1e-3)
    outer = np.maximum(med(r_pupil + .45 * (r_iris - r_pupil), r_pupil + .8 * (r_iris - r_pupil)), 1e-3)
    rim = np.maximum(med(r_pupil + .85 * (r_iris - r_pupil), r_pupil + .97 * (r_iris - r_pupil)), 1e-3)
    return dict(r_pupil=r_pupil, r_iris=r_iris, pupil_frac=r_pupil / r_iris, inner=inner.tolist(), outer=outer.tolist(),
                rim=rim.tolist(),
                near=med(r_iris + .1, r_iris + .22).tolist(), far=med(.86, .95).tolist(), source='ue_basecolor')


def build_eye(base: Path, side: str, out: Path | None = None, *, fast: bool = False, seed_salt: int = 0,
              ue_basecolor=None) -> dict:
    base = Path(base); out = Path(out or base)
    p = json.loads(_first(base, PARAMS[side]).read_text())
    try:
        src = _first(base, SOURCE[side])
        m = measure(np.asarray(Image.open(src).convert('RGB')))
    except FileNotFoundError:
        if not ue_basecolor:
            raise
        src, m = None, dict(pupil_frac=.26)          # base without angular textures (e.g. eye_map v2): default pupil
    if ue_basecolor:
        m = measure_basecolor_exr(ue_basecolor, m['pupil_frac'], G.ANATOMICAL['L'])
    if ue_basecolor:
        src = Path(ue_basecolor)
    d = G.dims_from_params(p, float(np.clip(m['pupil_frac'], .15, .55)))
    seed = int(p['structure'].get('seed', 1)) * 7919 + (0 if side == 'left' else 1) + 101 * seed_salt
    s = p['sclera']
    vasc = float(np.clip(.3 + .6 * float(s.get('vascularity_intensity', 0.)), 0, 1.5))   # MH range 0..2
    cov = float(np.clip(.3 + 1.2 * float(s.get('vascularity_coverage', .2)), 0, 1))
    ih, iw = (192, 1024) if fast else (384, 2048)
    sh, sw = (768, 2048) if fast else (1536, 4096)
    skirt = .9 * np.full(3, max(m['near'])) * np.array([.98, .97, .96])     # same neutral as the limbal white band
    ia, ih_ = T.iris_maps(T.IrisColors(np.array(m['inner']), np.array(m['outer']), d.pupil / d.L, skirt), seed,
                          h=ih, w=iw, structure=p.get('structure'))
    limbus = np.array(p['cornea'].get('limbus_color', (.85, .89, 1.)), float)
    sa, sh_, deep, sup, vessels = T.sclera_maps(
        d, T.ScleraColors(np.array(m['near']), np.array(m['far']), limbus), seed,
        vascularity=vasc, coverage=cov, h=sh, w=sw,
        periph_red=.25 if m.get('source') == 'ue_basecolor' else 1.)   # measured far colour already holds the redness
    amp = .00006 * d.scale
    parts = [G.shell(d), G.chamber(d), G.iris(d, ih_, amp), G.pupil_cup(d)]
    pre = f'eye2_{side}'
    Image.fromarray(linear_to_srgb_u8(ia)).save(out / f'{pre}_iris_albedo.png')
    u16(.5 + .5 * ih_).save(out / f'{pre}_iris_height.png')
    Image.fromarray(linear_to_srgb_u8(sa)).save(out / f'{pre}_sclera_albedo.png')
    u16(sh_).save(out / f'{pre}_sclera_height.png')
    arrays = {}
    for part in parts:
        arrays[f'{part.name}_positions'] = part.positions.astype(np.float32)
        arrays[f'{part.name}_triangles'] = part.triangles.astype(np.int32)
        arrays[f'{part.name}_uvs'] = part.uvs.astype(np.float32)
        if 'alpha' in part.extra:
            arrays[f'{part.name}_alpha'] = part.extra['alpha']
        if 'blend' in part.extra:
            arrays[f'{part.name}_blend'] = part.extra['blend']
    np.savez_compressed(out / f'{pre}.npz', **arrays)
    info = dict(version=ASSET_VERSION, side=side, source_texture=src.name if src else None,
                source_sha256=hashlib.sha256(src.read_bytes()).hexdigest() if src else None, params=_first(base, PARAMS[side]).name,
                measured=m, seed=seed, vascularity=vasc, coverage=cov,
                dims=dict(R=d.R, L=d.L, fade=d.fade, Rc=d.Rc, depth=d.depth, pupil=d.pupil, apex_z=d.apex_z, iris_z=d.iris_z,
                          z_limbus=d.z_limbus),
                iris_height_amp_m=amp, sclera_bump_m=.00008 * d.scale, ior=1.376,
                vessels=dict(count=len(vessels), crossings=T.crossings(vessels),
                             coverage_superficial=float((sup > .5).mean()), coverage_deep=float((deep > .3).mean())),
                fast=fast)
    (out / f'{pre}.json').write_text(json.dumps(info, indent=1))
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('base'); ap.add_argument('--side', default='both', choices=('left', 'right', 'both'))
    ap.add_argument('--out'); ap.add_argument('--fast', action='store_true')
    a = ap.parse_args()
    for side in (('left', 'right') if a.side == 'both' else (a.side,)):
        info = build_eye(Path(a.base), side, Path(a.out) if a.out else None, fast=a.fast)
        print(side, json.dumps(dict(measured=info['measured'], vessels=info['vessels'])))


if __name__ == '__main__':
    main()
