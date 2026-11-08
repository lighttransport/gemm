"""Layered eyeball meshes in the eye-local frame (metres; origin = eyeball centre, +Z = optical axis, +Y = up).

Dimensions come from the ``optics`` block of a vhuman eye parameter dict (``sclera_radius``, ``limbus_radius``,
``cornea_radius``, ``chamber_depth``). The cornea sphere is placed so that it meets the sclera sphere exactly at the
limbus circle; the iris plane sits ``chamber_depth`` behind the corneal apex.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Dims:
    R: float            # sclera (eyeball) radius
    L: float            # limbus radius (visible iris radius)
    Rc: float           # anterior corneal radius of curvature
    depth: float        # anterior chamber depth (apex -> iris plane)
    pupil: float        # pupil radius
    fade: float = .0011  # limbus feather width (metres at 12 mm scale)

    @property
    def z_limbus(self) -> float:
        return math.sqrt(self.R ** 2 - self.L ** 2)

    @property
    def theta_limbus(self) -> float:
        return math.asin(self.L / self.R)

    @property
    def cornea_center_z(self) -> float:
        return self.z_limbus - math.sqrt(self.Rc ** 2 - self.L ** 2)

    @property
    def apex_z(self) -> float:
        return self.cornea_center_z + self.Rc

    @property
    def iris_z(self) -> float:
        return self.apex_z - self.depth

    @property
    def scale(self) -> float:
        """Size relative to a 12 mm adult eyeball (vhuman heads are not always at anatomical scale)."""
        return self.R / .012

    def check(self):
        if not (0 < self.pupil < .6 * self.L < self.L < self.Rc < self.R):
            raise ValueError(f'inconsistent eye dims {self}')
        if not (self.z_limbus > self.iris_z > self.cornea_center_z):
            raise ValueError('iris plane outside the anterior chamber')
        return self


# Adult ratios to the eyeball radius (12 mm): limbus 5.85 mm (HVID 11.7), anterior corneal radius 7.8 mm,
# anterior chamber depth 3.0 mm.
# The visible iris diameter (HVID 11.7 mm) is measured through the cornea and includes its ~1.1x magnification; the
# true limbus is smaller; 5.1 mm matches UE silhouette-normalised iris edges on 020/100/038 left (2026-11-07).
ANATOMICAL = dict(L=5.1 / 12, Rc=7.8 / 12, depth=3.0 / 12)


def dims_from_params(p: dict, pupil_frac: float, *, anatomical: bool = True) -> Dims:
    """anatomical=True scales limbus/cornea/chamber from the eyeball radius with adult ratios. vhuman's optics
    block can pair a scaled eyeball (e.g. 14.6 mm) with an unscaled 6 mm limbus, which shrinks the iris."""
    o = p['optics']
    R = float(o['sclera_radius'])
    if anatomical:
        L, Rc, depth = (ANATOMICAL[k] * R for k in ('L', 'Rc', 'depth'))
    else:
        L, Rc, depth = float(o['limbus_radius']), float(o['cornea_radius']), float(o['chamber_depth'])
    # MH limbus_softness (UV units, .04 typical) widens the painted iris->sclera feather
    fade = min(.0014 + .012 * float(p.get('cornea', {}).get('limbus_softness', 0.)), .0028)
    return Dims(R=R, L=L, Rc=Rc, depth=depth, pupil=float(pupil_frac) * L, fade=fade).check()


@dataclass
class Part:
    name: str
    positions: np.ndarray            # (N, 3) float64
    triangles: np.ndarray            # (M, 3) int
    uvs: np.ndarray                  # (N, 2) float64, Blender convention (v up)
    smooth: bool = True
    extra: dict = field(default_factory=dict)


def _grid(rows: int, cols: int, first: int = 0, flip: bool = False) -> np.ndarray:
    """Triangles of a rows x (cols+1) vertex grid whose last column duplicates the first (UV seam)."""
    r = np.arange(rows - 1)[:, None]
    c = np.arange(cols)[None, :]
    w = cols + 1
    a = first + r * w + c
    b = a + 1
    d = a + w
    e = d + 1
    t = np.concatenate([np.stack([a, b, e], -1).reshape(-1, 3), np.stack([a, e, d], -1).reshape(-1, 3)])
    return t[:, ::-1].copy() if flip else t


def _phis(cols: int) -> np.ndarray:
    return np.linspace(0., 2 * math.pi, cols + 1)


# The sclera texture covers theta in [theta_limbus, pi] with v = s ** SCLERA_V_POWER, s the normalised polar angle,
# concentrating texels on the visible front.
SCLERA_V_POWER = .6


def sclera_v(theta, d: Dims):
    s = np.clip((np.asarray(theta) - d.theta_limbus) / (math.pi - d.theta_limbus), 0., 1.)
    return s ** SCLERA_V_POWER


def sclera_theta(v, d: Dims):
    return d.theta_limbus + np.clip(np.asarray(v), 0., 1.) ** (1. / SCLERA_V_POWER) * (math.pi - d.theta_limbus)


def sphere_dir(theta, phi):
    st = np.sin(theta)
    return np.stack([st * np.cos(phi), st * np.sin(phi), np.cos(theta)], -1)


LIMBUS_FADE = .0011          # metres at 12 mm scale: sclera fringe fading into the clear cornea (soft limbus)


def sclera(d: Dims, rings: int = 160, cols: int = 384, fade_rings: int = 12) -> Part:
    """Sclera sphere from the back up to LIMBUS_FADE inside the limbus. Inside the limbus the shell lies under the
    corneal surface (inside the cornea volume) and fades out through ``extra['alpha']`` (per vertex)."""
    # Rings dense near the limbus (where the silhouette and highlight live), sparse at the back.
    v = np.linspace(0., 1., rings)
    fade = LIMBUS_FADE * d.scale / d.R
    theta = np.concatenate([d.theta_limbus - fade * np.linspace(1., 0., fade_rings, endpoint=False), sclera_theta(v, d)])
    v = sclera_v(theta, d)
    rings = len(theta)
    phi = _phis(cols)
    T, PH = np.meshgrid(theta, phi, indexing='ij')
    pos = d.R * sphere_dir(T, PH).reshape(-1, 3)
    # texture row 0 (image top, Blender v = 1) is the back of the eye; the limbus is the bottom row (v = 0)
    uv = np.stack([PH / (2 * math.pi), np.broadcast_to(v[:, None], T.shape)], -1).reshape(-1, 2)
    # theta grows away from +Z: with u along phi this ordering faces outward after the flip below.
    tri = _grid(rings, cols, flip=True)
    t = np.clip((theta - (d.theta_limbus - fade)) / (1.3 * fade), 0., 1.)
    alpha = np.broadcast_to((t * t * (3 - 2 * t))[:, None], T.shape).reshape(-1)
    return Part('sclera', pos, tri, uv, extra=dict(alpha=alpha.astype(np.float32)))


def cornea(d: Dims, rings: int = 64, cols: int = 384) -> Part:
    """Closed aqueous/cornea volume: anterior cap + cylinder r=L + posterior disc behind the iris."""
    zc, a_l = d.cornea_center_z, math.asin(d.L / d.Rc)
    phi = _phis(cols)
    parts = []
    # cap (rings from apex to limbus); apex as a tiny ring avoids a fan
    a = np.linspace(0., a_l, rings)
    A, PH = np.meshgrid(a, phi, indexing='ij')
    cap = np.stack([d.Rc * np.sin(A) * np.cos(PH), d.Rc * np.sin(A) * np.sin(PH), zc + d.Rc * np.cos(A)], -1).reshape(-1, 3)
    parts.append((cap, _grid(rings, cols, flip=True)))
    z_b = d.iris_z - .25 * d.depth
    zs = np.linspace(d.z_limbus, z_b, 12)
    Z, PH = np.meshgrid(zs, phi, indexing='ij')
    cyl = np.stack([d.L * np.cos(PH), d.L * np.sin(PH), Z], -1).reshape(-1, 3)
    parts.append((cyl, _grid(len(zs), cols, flip=True)))
    rs = np.linspace(d.L, 0., 16)
    Rr, PH = np.meshgrid(rs, phi, indexing='ij')
    disc = np.stack([Rr * np.cos(PH), Rr * np.sin(PH), np.full_like(Rr, z_b)], -1).reshape(-1, 3)
    parts.append((disc, _grid(len(rs), cols, flip=True)))
    pos, tri, off = [], [], 0
    for p, t in parts:
        pos.append(p); tri.append(t + off); off += len(p)
    pos = np.concatenate(pos)
    uv = np.c_[.5 + pos[:, 0] / (2 * d.L), .5 + pos[:, 1] / (2 * d.L)]
    return Part('cornea', pos, np.concatenate(tri), uv, extra=dict(z_back=z_b))


# Iris texture v spans r in [pupil, pupil + IRIS_V_SPAN * (L - pupil)]; v = 1 / IRIS_V_SPAN is the limbus.
IRIS_V_SPAN = 1.25


def iris_profile(rn):
    """Axial offset (fraction of chamber depth) of the iris surface vs normalised radius (0 pupil, 1 limbus):
    pupil margin resting forward on the lens, a gentle collarette bulge, and the root dipping back."""
    rn = np.asarray(rn, float)
    return .07 * np.exp(-(rn / .12) ** 2) + .03 * np.exp(-((rn - .35) / .12) ** 2) - .06 * np.clip(rn - .85, 0, None) / .4


def iris(d: Dims, height: np.ndarray | None = None, amp: float = 0., rings: int = 112, cols: int = 720) -> Part:
    """Iris annulus from the pupil to the inner sclera wall, displaced along +Z by a polar height map
    (rows = v, cols = u, values in [-1, 1]) times ``amp`` metres."""
    r_wall = math.sqrt(d.R ** 2 - d.iris_z ** 2) * .995
    v = np.linspace(0., 1., rings) ** 1.15
    r = d.pupil + v * (r_wall - d.pupil)
    phi = _phis(cols)
    Rr, PH = np.meshgrid(r, phi, indexing='ij')
    rn = (Rr - d.pupil) / (d.L - d.pupil)
    z = d.iris_z + d.depth * iris_profile(rn)
    tv = np.clip(rn / IRIS_V_SPAN, 0., 1.)
    if height is not None and amp:
        h, w = height.shape
        yi = np.clip(np.rint(tv * (h - 1)).astype(int), 0, h - 1)
        xi = np.rint(PH / (2 * math.pi) * w).astype(int) % w
        z = z + amp * height[yi, xi] * np.clip(1.1 - rn, 0., 1.)
    pos = np.stack([Rr * np.cos(PH), Rr * np.sin(PH), z], -1).reshape(-1, 3)
    uv = np.stack([PH / (2 * math.pi), 1. - tv], -1).reshape(-1, 2)
    # faces +Z (toward the cornea)
    return Part('iris', pos, _grid(rings, cols, flip=True), uv)


def pupil_cup(d: Dims, rings: int = 16, cols: int = 96) -> Part:
    """Dark hemisphere behind the pupil, spanning to the lens equator, so the pupil reads as a hole."""
    rc = d.pupil * 1.6
    z0 = d.iris_z
    a = np.linspace(math.pi / 2, math.pi, rings)
    phi = _phis(cols)
    A, PH = np.meshgrid(a, phi, indexing='ij')
    pos = np.stack([rc * np.sin(A) * np.cos(PH), rc * np.sin(A) * np.sin(PH), z0 + .6 * rc * np.cos(A)], -1).reshape(-1, 3)
    uv = np.stack([PH / (2 * math.pi), np.broadcast_to(a[:, None], A.shape) / math.pi], -1).reshape(-1, 2)
    return Part('pupil_cup', pos, _grid(rings, cols, flip=False), uv)


def weld(pos: np.ndarray, tri: np.ndarray, tol: float = 1e-9):
    key = np.round(pos / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    t = inv.reshape(-1)[tri]
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    return pos[first], t


def boundary_edges(tri: np.ndarray) -> int:
    e = np.sort(np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), 1)
    _, c = np.unique(e, axis=0, return_counts=True)
    return int((c == 1).sum())


def face_normals(pos, tri):
    n = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)


# ------------------------------------------------------------------------------------- unified outer shell (v2)
LIMBUS_BLEND = .0007         # metres at 12 mm scale: smooth-max radius of the corneoscleral junction


def _smax(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0.) / k
    return np.maximum(a, b) + h * h * k * .25


def shell_radius(theta, d: Dims):
    """Distance from the eye centre to the outer surface: smooth union of the sclera and cornea spheres."""
    theta = np.asarray(theta, float)
    zc = d.cornea_center_z
    disc = d.Rc ** 2 - (zc * np.sin(theta)) ** 2
    t = np.where(disc > 0, zc * np.cos(theta) + np.sqrt(np.maximum(disc, 0.)), -1.)
    return _smax(np.full_like(theta, d.R), t, LIMBUS_BLEND * d.scale)


def shell(d: Dims, rings: int = 220, cols: int = 384) -> Part:
    """One smooth outer surface from the corneal apex to the back pole. extra['alpha'] = sclera opacity
    (0 over the clear cornea, 1 over the sclera, soft across the limbus); UVs follow the sclera map."""
    fade = d.fade * d.scale / d.R
    a0 = d.theta_limbus - .75 * fade
    th_front = np.linspace(0., a0, 40, endpoint=False)
    th_fade = np.linspace(a0, d.theta_limbus, 24, endpoint=False)
    th_back = sclera_theta(np.linspace(0., 1., rings), d)
    theta = np.concatenate([th_front, th_fade, th_back])
    rho = shell_radius(theta, d)
    phi = _phis(cols)
    T, PH = np.meshgrid(theta, phi, indexing='ij')
    pos = (rho[:, None, None] * sphere_dir(T, PH)).reshape(-1, 3)
    v = sclera_v(theta, d)
    uv = np.stack([PH / (2 * math.pi), np.broadcast_to(v[:, None], T.shape)], -1).reshape(-1, 2)
    def ss(x):
        x = np.clip(x, 0., 1.)
        return x * x * (3 - 2 * x)
    # UE-style painted limbus (non-physical): the shell turns opaque over the outer iris rim (alpha) and its colour
    # blends from the iris colour to the sclera texture across the limbus (blend), instead of revealing the recessed,
    # shadowed iris edge through clear cornea.
    alpha = np.broadcast_to(ss((theta - a0) / (.75 * fade))[:, None], T.shape).reshape(-1)
    blend = \
        np.broadcast_to(ss((theta - a0) / (1.35 * fade))[:, None], T.shape).reshape(-1)   # centred on the limbus
    return Part('shell', pos, _grid(len(theta), cols, flip=True), uv,
                extra=dict(alpha=alpha.astype(np.float32), blend=blend.astype(np.float32)))


def chamber(d: Dims, cols: int = 384) -> Part:
    """Posterior closure of the aqueous volume behind the clear cornea: cylinder r=L from the shell down to a disc
    behind the iris (normals out of the volume)."""
    th = math.asin(min(d.L / float(shell_radius(d.theta_limbus, d)), 1.))
    for _ in range(20):                                   # theta where rho(theta) sin(theta) = L
        th = math.asin(min(d.L / float(shell_radius(th, d)), 1.))
    z_top = float(shell_radius(th, d)) * math.cos(th)
    z_b = d.iris_z - .25 * d.depth
    phi = _phis(cols)
    zs = np.linspace(z_top, z_b, 12)
    Z, PH = np.meshgrid(zs, phi, indexing='ij')
    cyl = np.stack([d.L * np.cos(PH), d.L * np.sin(PH), Z], -1).reshape(-1, 3)
    rs = np.linspace(d.L, 0., 16)
    Rr, PH2 = np.meshgrid(rs, phi, indexing='ij')
    disc = np.stack([Rr * np.cos(PH2), Rr * np.sin(PH2), np.full_like(Rr, z_b)], -1).reshape(-1, 3)
    pos = np.concatenate([cyl, disc])
    tri = np.concatenate([_grid(len(zs), cols, flip=True), _grid(len(rs), cols, flip=True) + len(cyl)])
    uv = np.c_[.5 + pos[:, 0] / (2 * d.L), .5 + pos[:, 1] / (2 * d.L)]
    return Part('chamber', pos, tri, uv, extra=dict(z_top=z_top, z_back=z_b))
