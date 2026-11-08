"""Procedural iris (polar) and sclera (spherical) textures for the layered eye.

Iris map: rows = v (0 at the pupil margin, 1 at r = pupil + IRIS_V_SPAN * (limbus - pupil)), cols = azimuth.
Sclera map: rows = v (geometry.sclera_v of the polar angle, 0 at the limbus), cols = azimuth.
All colours are linear RGB. Image row 0 is the TOP of the PNG, i.e. Blender UV v = 1 - row/(H-1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..eye.noise import periodic_1d, rng, smoothstep, spectral, spectral_smooth
from . import geometry as G


# ----------------------------------------------------------------------------------------------- 3D noise
def wave_noise(dirs: np.ndarray, seed: int, *, k_lo: float, k_hi: float, n: int = 48, salt: int = 0) -> np.ndarray:
    """Seamless band-limited noise on unit directions: a sum of random plane waves (k in cycles per radian)."""
    g = rng(seed, 0x3D, salt)
    out = np.zeros(dirs.shape[:-1], np.float32)
    for _ in range(n):
        d = g.standard_normal(3); d /= np.linalg.norm(d)
        k = 2 * math.pi * math.exp(g.uniform(math.log(k_lo), math.log(k_hi)))
        out += (np.cos(k * (dirs @ d.astype(np.float32)) + g.uniform(0, 2 * math.pi)) / math.sqrt(k)).astype(np.float32)
    out -= out.mean()
    return out / (out.std() or 1.)


# ----------------------------------------------------------------------------------------------- iris
@dataclass
class IrisColors:
    inner: np.ndarray        # pupillary zone (inside the collarette), linear RGB
    outer: np.ndarray        # ciliary zone
    pupil_frac: float        # pupil radius / limbus radius
    skirt: np.ndarray | None = None   # colour of the hidden iris root beyond the limbus (seen through the soft veil)


def iris_maps(colors: IrisColors, seed: int, *, h: int = 384, w: int = 2048, structure: dict | None = None):
    """Returns (albedo (h,w,3) linear, height (h,w) in [-1,1]). ``structure`` takes the vhuman ``structure`` block
    (fibers / crypts / furrows / freckles / collarette, 0..1)."""
    s = {**dict(fibers=.5, crypts=.5, furrows=.5, freckles=.2, collarette=.5), **(structure or {})}
    g = rng(seed, 0x1715)
    v = (np.arange(h, dtype=np.float32) + .5) / h
    rn = (v * G.IRIS_V_SPAN)[:, None] * np.ones((1, w), np.float32)          # 0 pupil .. 1 limbus .. 1.25
    th = np.broadcast_to((np.arange(w, dtype=np.float32) / w)[None], (h, w))
    # collarette radius wanders with azimuth
    rc = .33 + .045 * periodic_1d(w, seed, beta=2.0, kmax=24, salt=1)[None]
    # radial stromal fibres: stretched along v, warped sideways so they wave
    warp = .004 * spectral_smooth((h, w), seed, beta=3., aniso=(1., 1.), kmax=24, salt=2)
    fib = spectral((h, w), seed, beta=.8, aniso=(1., 60.), kmax=1400, salt=3)
    cols = (np.arange(w)[None] + warp * w).astype(np.float32)
    fib = np.take_along_axis(fib, np.mod(np.rint(cols).astype(int), w), 1)
    fib_fine = spectral((h, w), seed, beta=.8, aniso=(1., 30.), kmax=1600, salt=4)
    # crypts of Fuchs: radially elongated lacunae either side of the collarette
    crypt = np.zeros((h, w), np.float32)
    for _ in range(int(10 + 50 * s['crypts'])):
        r0 = float(np.clip(g.normal(.38, .14), .12, .8)); t0 = g.uniform(0, 1)
        ar = g.uniform(.025, .07); at = g.uniform(.003, .008) / max(r0, .2)
        dt = np.mod(th - t0 + .5, 1.) - .5
        crypt = np.maximum(crypt, np.exp(-((rn - r0) / ar) ** 2 - (dt / at) ** 2) * g.uniform(.5, 1.))
    # contraction furrows: broken concentric arcs in the ciliary zone
    furrow = np.zeros((h, w), np.float32)
    for i in range(int(2 + 5 * s['furrows'])):
        r0 = g.uniform(.6, .95); wob = .012 * periodic_1d(w, seed, beta=2.5, kmax=12, salt=40 + i)[None]
        gate = smoothstep(.0, .6, periodic_1d(w, seed, beta=1.5, kmax=20, salt=60 + i))[None]
        furrow = np.maximum(furrow, np.exp(-((rn - r0 - wob) / .005) ** 2) * gate * gate)
    # freckles
    freck = np.zeros((h, w), np.float32)
    for _ in range(int(12 * s['freckles'])):
        r0 = g.uniform(.3, .9); t0 = g.uniform(0, 1); rad = g.uniform(.01, .025)
        dt = (np.mod(th - t0 + .5, 1.) - .5) * 2 * math.pi * r0
        freck = np.maximum(freck, smoothstep(rad, rad * .5, np.sqrt((rn - r0) ** 2 + dt ** 2)))
    coll = np.exp(-((rn - rc) / .03) ** 2)
    ruff = smoothstep(.06, .015, rn) * (.6 + .4 * np.clip(fib, -1, 1))             # pupillary ruff
    limbal = smoothstep(.82, 1.0, rn)                                               # dark limbal ring
    # height (relief): fibres + collarette ridge - crypts - furrows
    height = (.25 * s['fibers'] * fib + .15 * fib_fine + .9 * s['collarette'] * coll - 1.1 * crypt - .5 * furrow
              + .4 * ruff)
    height = np.clip(height / 1.6, -1., 1.).astype(np.float32)
    # albedo: zone colours modulated by structure (lighter ridges/fibres, dark crypts/furrows/limbal ring)
    zone = smoothstep(rc - .08, rc + .14, rn)[..., None]
    base = colors.inner * (1 - zone) + colors.outer * zone
    lum = (.55 * s['fibers'] * fib + .2 * fib_fine + .15 * s['collarette'] * coll - .8 * crypt - .25 * furrow
           - 1.0 * freck - 1.4 * ruff - .1 * limbal)
    albedo = base * np.exp(lum)[..., None]
    # fibre highlights desaturate toward a pale stromal colour
    pale = .5 * (base + base.mean(-1, keepdims=True))
    albedo = albedo + np.clip(.12 * fib, 0, None)[..., None] * (pale - albedo) * .5
    # calibrate: each zone's median equals the measured colour (the structure is not allowed to shift it)
    for mask, target in (((rn > .1) & (rn < rc - .03), colors.inner), ((rn > rc + .08) & (rn < .8), colors.outer)):
        med = np.median(albedo[mask], 0)
        gain = np.where(med > 1e-5, np.asarray(target) / np.maximum(med, 1e-5), 1.)
        sel = (zone[..., 0] < .5) if target is colors.inner else (zone[..., 0] >= .5)
        albedo[sel] *= gain
    albedo[rn > 1.02] = colors.skirt if colors.skirt is not None else colors.outer * .6   # hidden skirt, seen through the veil
    return np.clip(albedo, 0., 1.).astype(np.float32), height


# ----------------------------------------------------------------------------------------------- vessels
@dataclass
class Vessel:
    pts: np.ndarray          # (n, 3) unit directions
    width: np.ndarray        # (n,) metres
    deep: bool
    tree: int


class _Hash:
    def __init__(self, cell):
        self.cell, self.d = cell, {}

    def key(self, p):
        return tuple(np.floor(p / self.cell).astype(int))

    def add(self, a, b, tag):
        for p in (a, .5 * (a + b), b):
            self.d.setdefault(self.key(p), []).append((a, b, tag))

    def near(self, p):
        k = self.key(p)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    yield from self.d.get((k[0] + dx, k[1] + dy, k[2] + dz), ())


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-20), 0., 1.)
    return float(np.linalg.norm(p - (a + t * ab)))


def grow_vessels(d: G.Dims, seed: int, *, vascularity: float = .5, coverage: float = .5) -> list[Vessel]:
    """Arterial trees grown from the equator toward the limbus (deep episcleral trunks feeding superficial
    conjunctival branches). Tips stop before touching any other vessel, so the result is a forest whose
    drawn paths never cross (no visual loops)."""
    g = rng(seed, 0x7E55)
    R, k = d.R, d.scale
    step = .00022 * k / R                                    # radians
    clear = .00012 * k / R
    th_stop = d.theta_limbus + .0004 * k / R
    hsh = _Hash(.0012 * k / R)
    vessels: list[Vessel] = []
    w0 = .0001 * k * (.8 + .5 * vascularity)
    n_trunks = int(round(8 + 16 * coverage))
    stack = []
    phis = (np.arange(n_trunks) + g.uniform(0, 1, n_trunks) * .8) / n_trunks * 2 * math.pi + g.uniform(0, 2 * math.pi)
    for i, ph in enumerate(phis):
        th = math.radians(g.uniform(80, 100))
        p = G.sphere_dir(th, ph)
        etheta = np.array([math.cos(th) * math.cos(ph), math.cos(th) * math.sin(ph), -math.sin(th)])
        stack.append((p, -etheta, w0 * g.uniform(.7, 1.1), i, 0, -1))
    w_min = .000024 * k
    budget = 6000 + int(9000 * coverage)
    segs = 0
    while stack and segs < budget:
        p, t, w, tree, gen, parent = stack.pop()
        pts, ws = [p], [w]
        curl = g.normal(0, .25)
        for n_step in range(400):
            th = math.acos(np.clip(p[2], -1, 1)); ph = math.atan2(p[1], p[0])
            to_limbus = -np.array([math.cos(th) * math.cos(ph), math.cos(th) * math.sin(ph), -math.sin(th)])
            curl = .6 * curl + g.normal(0, .45)
            side = np.cross(p, t)
            pull = .16 if gen == 0 else .045
            t = t + pull * (to_limbus - t) + .3 * curl * side
            t -= np.dot(t, p) * p; t /= np.linalg.norm(t)
            q = p + step * t; q /= np.linalg.norm(q)
            # stop before touching another vessel (any vessel but our own last few points)
            hit = False
            for a, b, tag in hsh.near(q):
                if tag in (parent, parent + 1000000) and min(np.linalg.norm(a - pts[0]), np.linalg.norm(b - pts[0])) < 1.5 * step:
                    continue                              # only the parent segments at the branch point itself
                if _seg_dist(q, a, b) < clear + w / R:
                    hit = True; break
            if hit or math.acos(np.clip(q[2], -1, 1)) < th_stop:
                break
            if len(pts) >= 4:                                 # own trail, lagged so the tip never sees itself
                hsh.add(pts[-4], pts[-3], len(vessels) + 1000000)
            pts.append(q); ws.append(w); p = q; segs += 1
            near_limbus = math.acos(np.clip(p[2], -1, 1)) < d.theta_limbus + .003 * k / R
            p_branch = (.09 if not near_limbus else .16) * (1 + vascularity)
            if g.uniform() < p_branch and w > 1.4 * w_min:
                a = g.uniform(.6, .9)                             # Murray: w^3 = w1^3 + w2^3
                w1 = w * (a ** 3 / (a ** 3 + (1 - a ** 3))) ** (1 / 3)
                w2 = (max(w ** 3 - w1 ** 3, 0.)) ** (1 / 3)
                ang = math.radians(g.uniform(30, 75)) * (1 if g.uniform() < .5 else -1)
                side = np.cross(p, t)
                tb = math.cos(ang) * t + math.sin(ang) * side
                stack.append((p.copy(), tb, w2, tree, gen + 1, len(vessels)))
                w = w1
            w *= .997
            if w < w_min:
                break
        if len(pts) >= 3:
            for a, b in zip(pts[:-1], pts[1:]):
                hsh.add(a, b, len(vessels))
            vessels.append(Vessel(np.array(pts), np.array(ws), deep=gen == 0, tree=tree))
    # conjunctival capillary network: short, thin, tortuous vessels seeded in the periphery, same no-contact rule
    n_cap = int((700 + 1800 * coverage) * (.5 + vascularity))
    for c in range(n_cap):
        parent = -1
        if vessels and g.uniform() < .75:
            # sprout from a point on an existing vessel (reticular network rather than isolated squiggles)
            parent = int(g.integers(len(vessels))); pv = vessels[parent]
            i = int(g.integers(1, len(pv.pts) - 1)); p = pv.pts[i].copy()
            tang = pv.pts[i + 1] - pv.pts[i - 1]; tang -= np.dot(tang, p) * p; tang /= max(np.linalg.norm(tang), 1e-12)
            ang = math.radians(g.uniform(45, 90)) * (1 if g.uniform() < .5 else -1)
            t = math.cos(ang) * tang + math.sin(ang) * np.cross(p, tang)
            if math.acos(np.clip(p[2], -1, 1)) < th_stop + 2 * step:
                continue
        else:
            th0 = d.theta_limbus + (math.radians(g.uniform(12, 70)) if g.uniform() < .8 else math.radians(g.uniform(4, 12)))
            ph0 = g.uniform(0, 2 * math.pi)
            p = G.sphere_dir(th0, ph0)
            a = g.uniform(0, 2 * math.pi)
            e1 = np.cross(p, [0., 0., 1.]); e1 /= np.linalg.norm(e1); e2 = np.cross(p, e1)
            t = math.cos(a) * e1 + math.sin(a) * e2
        w = w_min * g.uniform(.6, 1.3)
        pts, ws, curl = [p], [w], 0.
        if parent >= 0:                                   # first step straight out of the parent, clear of it
            q = p + 1.6 * step * t; q /= np.linalg.norm(q)
            mid = .5 * (p + q); mid /= np.linalg.norm(mid)
            if any(_seg_dist(x, a_, b_) < clear + w / R for x in (q, mid) for a_, b_, tag in hsh.near(x)
                   if not (tag in (parent, parent + 1000000) and
                           min(np.linalg.norm(a_ - pts[0]), np.linalg.norm(b_ - pts[0])) < 1.5 * step)):
                continue
            pts.append(q); ws.append(w); p = q
        for n_step in range(int(g.uniform(6, 30))):
            curl = .5 * curl + g.normal(0, .55)
            t = t + .45 * curl * np.cross(p, t); t -= np.dot(t, p) * p; t /= np.linalg.norm(t)
            q = p + .6 * step * t; q /= np.linalg.norm(q)
            if math.acos(np.clip(q[2], -1, 1)) < th_stop or any(
                    _seg_dist(q, a_, b_) < clear + w / R for a_, b_, tag in hsh.near(q)
                    if not (tag in (parent, parent + 1000000) and
                            min(np.linalg.norm(a_ - pts[0]), np.linalg.norm(b_ - pts[0])) < 1.5 * step)):
                break
            if len(pts) >= 3:
                hsh.add(pts[-3], pts[-2], len(vessels) + 1000000)
            pts.append(q); ws.append(w); p = q
        if len(pts) >= 4:
            for a_, b_ in zip(pts[:-1], pts[1:]):
                hsh.add(a_, b_, len(vessels))
            vessels.append(Vessel(np.array(pts), np.array(ws), deep=False, tree=-1 - c))
    return vessels


def crossings(vessels: list[Vessel]) -> int:
    """Count pairs of non-adjacent segments that intersect (in the tangent plane), across all vessels.
    Branch roots share a point with their parent and are not counted."""
    segs = []
    for vi, v in enumerate(vessels):
        for i in range(len(v.pts) - 1):
            segs.append((v.pts[i], v.pts[i + 1], vi, i))
    if not segs:
        return 0
    A = np.array([s[0] for s in segs]); B = np.array([s[1] for s in segs])
    mid = .5 * (A + B)
    cell = np.max(np.linalg.norm(B - A, axis=1)) * 2 + 1e-9
    keys = {}
    for i, m in enumerate(mid):
        keys.setdefault(tuple(np.floor(m / cell).astype(int)), []).append(i)
    n = 0
    for i, m in enumerate(mid):
        k = np.floor(m / cell).astype(int)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for j in keys.get((k[0] + dx, k[1] + dy, k[2] + dz), ()):
                        if j <= i or (segs[i][2] == segs[j][2] and abs(segs[i][3] - segs[j][3]) <= 1):
                            continue
                        if min(np.linalg.norm(A[i] - A[j]), np.linalg.norm(A[i] - B[j]), np.linalg.norm(B[i] - A[j]),
                               np.linalg.norm(B[i] - B[j])) < 1e-12:
                            continue                       # shared branch point
                        nrm = m / np.linalg.norm(m)
                        e1 = np.cross(nrm, [0., 0., 1.]) if abs(nrm[2]) < .9 else np.cross(nrm, [1., 0., 0.])
                        e1 /= np.linalg.norm(e1); e2 = np.cross(nrm, e1)
                        P = lambda x: np.array([x @ e1, x @ e2])
                        a, b, c, d = P(A[i]), P(B[i]), P(A[j]), P(B[j])
                        def orient(p, q, r):
                            return np.sign((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
                        if orient(a, b, c) * orient(a, b, d) < 0 and orient(c, d, a) * orient(c, d, b) < 0:
                            n += 1
    return n


# ----------------------------------------------------------------------------------------------- sclera
@dataclass
class ScleraColors:
    near: np.ndarray          # just outside the limbus (linear RGB)
    far: np.ndarray           # periphery (equator)
    limbus: np.ndarray        # limbal tint (multiplier)


def sclera_dirs(d: G.Dims, h: int, w: int) -> np.ndarray:
    v = 1. - (np.arange(h, dtype=np.float64) + .5) / h        # row 0 = top = v 1 (back of the eye)
    th = G.sclera_theta(v, d)
    ph = (np.arange(w, dtype=np.float64) + .5) / w * 2 * math.pi
    T, P = np.meshgrid(th, ph, indexing='ij')
    return G.sphere_dir(T, P).astype(np.float32), T


def rasterize(vessels: list[Vessel], d: G.Dims, dirs: np.ndarray, *, deep: bool, soft: float) -> np.ndarray:
    """Coverage mask: gaussian cross-section of width w (deep vessels blurred by ``soft``)."""
    h, w = dirs.shape[:2]
    out = np.zeros((h, w), np.float32)
    R = d.R
    for vs in vessels:
        if vs.deep != deep:
            continue
        for i in range(len(vs.pts) - 1):
            a, b = vs.pts[i], vs.pts[i + 1]
            wid = .5 * (vs.width[i] + vs.width[i + 1])
            sig_true = (.5 * wid * (1 + soft)) / R
            th_m = math.acos(np.clip(.5 * (a[2] + b[2]), -1, 1))
            texel = min(2 * math.pi * max(math.sin(th_m), .05) / w,
                        (math.pi - d.theta_limbus) / h / max(G.SCLERA_V_POWER * max(float(G.sclera_v(th_m, d)), .05) ** ((G.SCLERA_V_POWER - 1) / G.SCLERA_V_POWER) / 1., 1e-9))
            sig = max(sig_true, .6 * texel)
            amp = sig_true / sig
            reach = 3 * sig + np.linalg.norm(b - a)
            m = .5 * (a + b); m /= np.linalg.norm(m)
            th = math.acos(np.clip(m[2], -1, 1)); ph = math.atan2(m[1], m[0]) % (2 * math.pi)
            vv = float(G.sclera_v(th, d)); row = (1 - vv) * h - .5
            dth = reach; dv = abs(float(G.sclera_v(th + dth, d)) - float(G.sclera_v(max(th - dth, d.theta_limbus), d)))
            r0, r1 = int(max(row - dv * h - 2, 0)), int(min(row + dv * h + 3, h))
            dc = reach / max(math.sin(th), 1e-3) / (2 * math.pi) * w
            c0, c1 = int(math.floor(ph / (2 * math.pi) * w - dc - 2)), int(math.ceil(ph / (2 * math.pi) * w + dc + 3))
            cols = np.arange(c0, c1) % w
            P = dirs[r0:r1][:, cols]
            ab = (b - a).astype(np.float32)
            t = np.clip(((P - a.astype(np.float32)) @ ab) / max(float(ab @ ab), 1e-20), 0, 1)
            dist = np.linalg.norm(P - (a.astype(np.float32) + t[..., None] * ab), axis=-1)
            prof = amp * (np.exp(-.5 * (dist / sig) ** 2) if soft or amp < 1 else smoothstep(1.25 * sig, .75 * sig, dist))
            sub = out[r0:r1][:, cols]
            out[r0:r1, cols] = np.maximum(sub, prof.astype(np.float32))
    return out


def sclera_maps(d: G.Dims, colors: ScleraColors, seed: int, *, vascularity: float = .5, coverage: float = .5,
                h: int = 1536, w: int = 4096, vessels: list[Vessel] | None = None, periph_red: float = 1.):
    """Returns (albedo (h,w,3) linear, height (h,w) in [0,1], deep mask, superficial mask)."""
    dirs, T = sclera_dirs(d, h, w)
    if vessels is None:
        vessels = grow_vessels(d, seed, vascularity=vascularity, coverage=coverage)
    deep = rasterize(vessels, d, dirs, deep=True, soft=1.5)
    sup = rasterize(vessels, d, dirs, deep=False, soft=0.)
    ang = (T - d.theta_limbus) * d.R / d.scale               # arc distance from the limbus at 12 mm scale (m)
    # base: near -> far colour plus vascular redness toward the fornix
    f = smoothstep(.0015, .011, ang)[..., None]
    base = colors.near * (1 - f) + colors.far * f
    cloud = wave_noise(dirs, seed, k_lo=2., k_hi=9., salt=1)          # whitish/pink cloudy patches (~2-6 mm)
    mott = wave_noise(dirs, seed, k_lo=10., k_hi=30., salt=2)         # fine mottling (~0.5-1.5 mm)
    base = base * (1 + .05 * cloud + .025 * mott)[..., None]
    # whitish, less vascular patches (Tenon's/conjunctival thickening) read as pale clouds over the pink
    pale = (np.clip(cloud - .3, 0, None) * (1 - f[..., 0]))[..., None]
    base = base + .2 * pale * (base * 1.12 - base)
    # conjunctival/episcleral redness toward the fornix (MetaHuman-style peripheral tint; same for every identity)
    fr = smoothstep(.0055, .0115, ang) * np.clip(.85 + .15 * cloud, .5, 1.1) * (.7 + .6 * vascularity)
    red = np.array([1., .6, .58], np.float32)
    base = base * (1 - np.clip(periph_red * fr, 0, 1.2)[..., None] * (1 - red))
    # limbal transition: grey shadowed band (~1.5 mm) where the sclera thins over the ciliary body
    # canthal redness: the exposed nasal/temporal conjunctiva (horizontal, away from the limbus) is redder (UE look)
    horiz = (dirs[..., 0] ** 2 / np.maximum(1 - dirs[..., 2] ** 2, 1e-6)) ** 2
    canth = (horiz * smoothstep(.003, .009, ang) * (.6 + .5 * vascularity))[..., None]
    base = base * (1 - .35 * canth * (1 - red))
    lb = smoothstep(.0014, .0002, ang)[..., None]
    wb = smoothstep(.004, .0006, ang)[..., None]
    # UE renders the near-limbus sclera grey-white even on red eyes; non-physical neutral band to match (the soft
    # limbus veil takes this colour, so a red band here reads as a red ring around the iris)
    white = np.maximum(base.max(-1, keepdims=True), .3) * np.array([.98, .97, .96], np.float32)
    base = base + .85 * wb * (white - base)
    # vessels: deep = bluish-purple blurred, superficial = crisp red
    deep_c = np.array([.8, .42, .42], np.float32); sup_c = np.array([.66, .08, .06], np.float32)
    vis = (.35 + .65 * smoothstep(.0008, .005, ang))[..., None]
    alb = base * (1 - .7 * vis * deep[..., None] * (1 - deep_c)) * (1 - np.clip(vis * sup[..., None], 0, 1) * (1 - sup_c))
    # surface height: tear film / conjunctival undulation (mm scale) + slightly raised superficial vessels
    hgt = .6 * wave_noise(dirs, seed, k_lo=6., k_hi=22., salt=3) + .25 * mott + .12 * sup
    hgt = (hgt - hgt.min()) / max(float(np.ptp(hgt)), 1e-6)
    return np.clip(alb, 0, 1).astype(np.float32), hgt.astype(np.float32), deep, sup, vessels
