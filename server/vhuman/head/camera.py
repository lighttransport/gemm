"""Pixal3D's single-view camera, from portrait pixels to rays in the GLB frame.

Pixal3D is pixel-aligned: it projects its voxel grid into the input image
with a pinhole camera (cpu/pixal3d/math.cc, pixal3d_project):
- grid point p in [-0.5, 0.5]^3 / mesh_scale;
- the camera at p = (0, 0, d), d = 0.5 / (tan(fov/2) * mesh_scale),
  looking along -p.z;
- image x = f p.x / (d - p.z) + W/2, y = -f p.y / (d - p.z) + H/2, with
  f = W / (2 tan(fov/2)).
The image is Pixal3D's preprocessed crop (cpu/pixal3d/preprocess.cc): the
input is downscaled to at most 1024, cropped to the alpha > 204 bounding
box grown 1.1x into a square, then resized.
The GLB stores (-p.x, p.y, -p.z) (cpu/pixal3d/postprocess.cc), so the face
the camera saw looks along -Z.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass
class PixalCamera:
    fov: float                 # radians
    scale: float               # input downscale (<= 1)
    left: float                # crop origin in the (downscaled) input, pixels
    top: float
    side: float                # crop size, pixels
    mesh_scale: float = 1.0

    @property
    def distance(self) -> float:
        return 0.5 / (math.tan(self.fov / 2) * self.mesh_scale)

    @property
    def focal(self) -> float:
        return self.side * 0.5 / math.tan(self.fov / 2)

    @property
    def origin(self) -> np.ndarray:
        """The camera centre in GLB coordinates."""
        return np.array([0.0, 0.0, -self.distance])

    @classmethod
    def from_portrait(cls, portrait, fov: float, mesh_scale: float = 1.0) -> "PixalCamera":
        img = Image.open(portrait) if not isinstance(portrait, Image.Image) else portrait
        rgba = img.convert("RGBA")
        w, h = rgba.size
        scale = 1.0
        if max(w, h) > 1024:
            scale = 1024.0 / max(w, h)
            rgba = rgba.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        a = np.asarray(rgba)[..., 3] > 204
        ys, xs = np.nonzero(a)
        if len(xs) == 0:
            raise ValueError("the portrait has no foreground with alpha > 0.8")
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        extent = int(max(x1 - x0, y1 - y0) * 1.1)
        half = extent // 2
        side = 2 * half
        left = int(round((x0 + x1) * 0.5 - half))
        top = int(round((y0 + y1) * 0.5 - half))
        return cls(fov, scale, float(left), float(top), float(side), mesh_scale)

    def to_crop(self, x, y):
        """Portrait pixel coordinates -> crop pixel coordinates."""
        return np.asarray(x, np.float64) * self.scale - self.left, np.asarray(y, np.float64) * self.scale - self.top

    def rays(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Rays (origin, unit direction) in the GLB frame through portrait pixels."""
        cx, cy = self.to_crop(x, y)
        a = (cx - self.side / 2) / self.focal
        b = -(cy - self.side / 2) / self.focal
        d = np.stack([-a, b, np.ones_like(a)], -1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        return np.broadcast_to(self.origin, d.shape), d

    def project(self, points: np.ndarray) -> np.ndarray:
        """GLB points (..., 3) -> portrait pixel coordinates (..., 2)."""
        p = np.asarray(points, np.float64) * np.array([-1.0, 1.0, -1.0])     # back to the grid frame
        depth = self.distance - p[..., 2]
        cx = self.focal * p[..., 0] / depth + self.side / 2
        cy = -self.focal * p[..., 1] / depth + self.side / 2
        return np.stack([(cx + self.left) / self.scale, (cy + self.top) / self.scale], -1)

    def as_dict(self) -> dict:
        return {"fov_deg": math.degrees(self.fov), "scale": self.scale, "left": self.left, "top": self.top,
                "side": self.side, "distance": self.distance, "focal": self.focal}


def ray_mesh(origin: np.ndarray, direction: np.ndarray, v0, e1, e2, eps: float = 1e-12):
    """Nearest hit of one ray against all triangles (Moller-Trumbore);
    returns (t, triangle index) or (inf, -1). v0, e1, e2 are (T, 3)."""
    pvec = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, pvec)
    ok = np.abs(det) > eps
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tvec = origin - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv
    qvec = np.cross(tvec, e1)
    v = (qvec @ direction) * inv
    t = np.einsum("ij,ij->i", e2, qvec) * inv
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    if not hit.any():
        return math.inf, -1
    t = np.where(hit, t, np.inf)
    k = int(np.argmin(t))
    return float(t[k]), k


def mesh_from_glb(path: Path):
    """Positions, normals, UVs, triangles and the raw GLB of a Pixal3D mesh."""
    from ..eye.glb import GLB
    g = GLB.load(path)
    prim = g.doc["meshes"][0]["primitives"][0]
    a = prim["attributes"]
    return {"glb": g, "positions": g.accessor(a["POSITION"]).astype(np.float64),
            "normals": g.accessor(a["NORMAL"]).astype(np.float64),
            "uvs": g.accessor(a["TEXCOORD_0"]).astype(np.float64),
            "triangles": g.accessor(prim["indices"]).reshape(-1, 3).astype(np.int64),
            "material": g.doc["materials"][prim.get("material", 0)]}
