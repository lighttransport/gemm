"""Analytic eye meshes: a smooth union of two spheres over a recessed iris.

The shell is closed and uses an independently designed angular UV map. The
portable GLB has a transparent corneal region and an opaque iris disk."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import optics
from . import params as P


@dataclass
class Mesh:
    name: str
    positions: np.ndarray   # (N, 3) float32
    normals: np.ndarray     # (N, 3) float32
    uvs: np.ndarray         # (N, 2) float32 (glTF: v down)
    indices: np.ndarray     # (M, 3) uint32
    tangents: np.ndarray | None = None   # (N, 4) float32

    def with_tangents(self) -> "Mesh":
        self.tangents = compute_tangents(self.positions, self.normals, self.uvs, self.indices)
        return self


def _grid_indices(rings: int, segments: int, first: int) -> np.ndarray:
    """Quads between consecutive rings of `segments` vertices (ring-major,
    starting at vertex `first`), wrapping around the angle."""
    r = np.arange(rings - 1)[:, None]
    s = np.arange(segments)[None, :]
    a = first + r * segments + s
    b = first + r * segments + (s + 1) % segments
    c = a + segments
    d = b + segments
    return np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])


def shell_alphas(profile: optics.Profile, rings: int) -> np.ndarray:
    """Ring angles, dense over the cornea and the limbus."""
    a_l = profile.alpha_limbus
    front = int(rings * 0.6)
    near = np.linspace(0.0, 1.35 * a_l, front + 1)[1:]
    back = np.linspace(1.35 * a_l, math.pi, rings - front + 1)[1:-1]
    return np.concatenate([near, back])


def shell(profile: optics.Profile = optics.ANATOMICAL, rings: int = 120, segments: int = 160) -> Mesh:
    alphas = shell_alphas(profile, rings)
    phis = np.linspace(0.0, 2 * math.pi, segments, endpoint=False)
    a, f = np.meshgrid(alphas, phis, indexing="ij")
    dirs = np.stack([np.sin(a) * np.cos(f), np.sin(a) * np.sin(f), np.cos(a)], -1).reshape(-1, 3)
    rho = optics.surface_radius(a.reshape(-1), profile)
    # Analytic normal of r = rho(alpha): n ~ r_hat - (rho'/rho) alpha_hat.
    h = 1e-5
    drho = (optics.surface_radius(a.reshape(-1) + h, profile) - optics.surface_radius(a.reshape(-1) - h, profile)) / (2 * h)
    a_hat = np.stack([np.cos(a) * np.cos(f), np.cos(a) * np.sin(f), -np.sin(a)], -1).reshape(-1, 3)
    normals = optics.normalize(dirs - (drho / rho)[:, None] * a_hat)
    apex = np.array([[0.0, 0.0, 1.0]])
    pole = np.array([[0.0, 0.0, -1.0]])
    positions = np.concatenate([apex * profile.apex_z, dirs * rho[:, None], pole * profile.sclera_radius])
    normals = np.concatenate([apex, normals, pole])
    uvs = np.concatenate([[[0.5, 0.5]], optics.eyeball_uv(dirs, profile),
                          [[0.5, 0.5 + optics.BACK_UV_RADIUS]]])
    n_ring = len(alphas)
    first, last = 1, 1 + n_ring * segments
    s = np.arange(segments)
    apex_fan = np.stack([np.zeros(segments, int), first + s, first + (s + 1) % segments], -1)
    last_ring = first + (n_ring - 1) * segments
    pole_fan = np.stack([np.full(segments, last), last_ring + (s + 1) % segments, last_ring + s], -1)
    indices = np.concatenate([apex_fan, _grid_indices(n_ring, segments, first), pole_fan]).astype(np.uint32)
    return Mesh("eyeball_shell", positions.astype(np.float32), normals.astype(np.float32),
                uvs.astype(np.float32), indices).with_tangents()


def iris_disk(p: dict, profile: optics.Profile | None = None, rings: int = 64,
              segments: int = 160) -> Mesh:
    """The iris at the iris plane (radius = limbus), plus the dark wall up
    towards the limbus plane. UVs are the iris texture layout."""
    p = P.validate(p)
    profile = profile or optics.profile_from_params(p)
    o = p["optics"]
    r_l, r_iris = profile.limbus_radius, optics.iris_radius(p, profile)
    z_iris = profile.iris_z(o["chamber_depth"])
    phis = np.linspace(0.0, 2 * math.pi, segments, endpoint=False)
    radii = np.linspace(0.0, r_l, rings + 1)[1:]
    wall_top = min(profile.z_limbus - 0.0002, z_iris + 0.0015)
    wall = np.linspace(z_iris, wall_top, 5)[1:]
    rr, ff = np.meshgrid(radii, phis, indexing="ij")
    z_disk = z_iris + o["iris_convexity"] * (1.0 - rr / r_l)
    disk = np.stack([rr * np.cos(ff), rr * np.sin(ff), z_disk], -1).reshape(-1, 3)
    ww, fw = np.meshgrid(wall, phis, indexing="ij")
    wallp = np.stack([np.full_like(ww, r_l) * np.cos(fw), np.full_like(ww, r_l) * np.sin(fw), ww], -1).reshape(-1, 3)
    positions = np.concatenate([[[0.0, 0.0, z_iris + o["iris_convexity"]]], disk, wallp])
    slope = -o["iris_convexity"] / r_l
    n_disk = optics.normalize(np.stack([-slope * np.cos(ff), -slope * np.sin(ff), np.ones_like(ff)], -1).reshape(-1, 3))
    n_wall = np.stack([-np.cos(fw), -np.sin(fw), np.zeros_like(fw)], -1).reshape(-1, 3)
    normals = np.concatenate([[[0.0, 0.0, 1.0]], n_disk, n_wall])
    # Texture radius: the disk's t = r / r_iris, then the wall continues outwards.
    t_disk = rr / r_iris
    t_wall = (r_l + (ww - z_iris)) / r_iris
    t = np.concatenate([[0.0], t_disk.reshape(-1), t_wall.reshape(-1)])
    ang = np.concatenate([[0.0], ff.reshape(-1), fw.reshape(-1)])
    s = optics.IRIS_TEX_SCALE
    uvs = np.stack([0.5 + s * t * np.cos(ang), 0.5 - s * t * np.sin(ang)], -1)
    k = np.arange(segments)
    fan = np.stack([np.zeros(segments, int), 1 + k, 1 + (k + 1) % segments], -1)   # faces +Z
    quads = _grid_indices(rings + len(wall), segments, 1)       # disk faces +Z, the wall faces the axis
    indices = np.concatenate([fan, quads]).astype(np.uint32)
    return Mesh("iris", positions.astype(np.float32), normals.astype(np.float32), uvs.astype(np.float32),
                indices).with_tangents()


def compute_tangents(pos, nrm, uv, idx) -> np.ndarray:
    """Per-vertex tangents (glTF: xyz + handedness w) from UV derivatives."""
    tri = idx.astype(np.int64)
    p0, p1, p2 = (pos[tri[:, k]].astype(np.float64) for k in range(3))
    w0, w1, w2 = (uv[tri[:, k]].astype(np.float64) for k in range(3))
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = w1 - w0, w2 - w0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(det) > 1e-20, 1.0 / np.where(det == 0, 1, det), 0.0)[:, None]
    sdir = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r
    tdir = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r
    tan = np.zeros((len(pos), 3))
    bit = np.zeros((len(pos), 3))
    for k in range(3):
        np.add.at(tan, tri[:, k], sdir)
        np.add.at(bit, tri[:, k], tdir)
    n = nrm.astype(np.float64)
    t = tan - n * np.sum(n * tan, -1, keepdims=True)
    bad = np.linalg.norm(t, axis=-1) < 1e-12
    # Degenerate (apex, pole): any vector perpendicular to the normal.
    fallback = np.cross(n, np.array([0.0, 1.0, 0.0]))
    fallback[np.linalg.norm(fallback, axis=-1) < 1e-6] = [1.0, 0.0, 0.0]
    t[bad] = fallback[bad]
    t = optics.normalize(t)
    w = np.where(np.sum(np.cross(n, t) * bit, -1) < 0.0, -1.0, 1.0)
    return np.concatenate([t, w[:, None]], -1).astype(np.float32)


def edge_use_counts(indices: np.ndarray) -> np.ndarray:
    """How many triangles use each undirected edge (closed manifold: all 2)."""
    tri = indices.astype(np.int64)
    edges = np.concatenate([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]])
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return counts
