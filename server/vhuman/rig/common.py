"""The head frame, subject meshes and small geometry helpers shared by the rig.

The rig works in a *head frame* H, in metres: +Y up, the face looks along
+Z, +X is the subject's left, and the origin is the midpoint of the two
eyeball centres. A fitted head (head/fit.py) lives in Pixal3D's GLB frame,
where the face looks along -Z at `units_per_m` units per metre:
    p_H = S (p_glb - o) / k,   S = diag(-1, 1, -1)   (a rotation about Y).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..eye.glb import GLB

FLIP = np.array([-1.0, 1.0, -1.0])


@dataclass
class Frame:
    origin: np.ndarray        # GLB point that maps to the H origin
    k: float                  # GLB units per metre

    def to_h(self, p) -> np.ndarray:
        return (np.asarray(p, np.float64) - self.origin) * FLIP / self.k

    def to_glb(self, p) -> np.ndarray:
        return np.asarray(p, np.float64) * FLIP * self.k + self.origin

    def dir_to_h(self, d) -> np.ndarray:
        return np.asarray(d, np.float64) * FLIP

    def rot_to_h(self, r) -> np.ndarray:
        """An orientation (columns = the local axes in GLB coordinates) in H:
        the same axes expressed in H, i.e. S R (a proper rotation: det S = 1)."""
        return np.diag(FLIP) @ np.asarray(r, np.float64)

    def as_dict(self) -> dict:
        return {"origin_glb": self.origin.round(7).tolist(), "units_per_m": self.k,
                "axes": "H = diag(-1,1,-1) (glb - origin) / units_per_m"}


@dataclass
class Subject:
    """A fitted head folder read back: its frame, head surface (H frame), eye
    poses (H frame) and the files needed for texture transfer."""
    folder: Path
    frame: Frame
    positions: np.ndarray     # (V, 3) head surface, H frame (carved and draped lids)
    normals: np.ndarray
    uvs: np.ndarray
    tangents: np.ndarray      # (V, 4) xyz in H, w = handedness
    triangles: np.ndarray
    eyes: list                # [{side, center, rotation (3x3, columns x y gaze), radius}] in H
    fit: dict                 # fit.json
    glb: GLB                  # head_eyes.glb

    @property
    def portrait(self) -> Path:
        return self.folder / "portrait.png"


def load_subject(folder) -> Subject:
    from ..eye import optics
    folder = Path(folder)
    fit = json.loads((folder / "fit.json").read_text())
    eyes_glb = fit["export"]["eyes"]
    k = float(fit["fit"]["units_per_m"])
    centres = np.array([e["center"] for e in eyes_glb], np.float64)
    frame = Frame(centres.mean(0), k)
    g = GLB.load(folder / "head_eyes.glb")
    mesh = next(m for m in g.doc["meshes"] if m["name"] == "head")
    prim = mesh["primitives"][0]
    a = prim["attributes"]
    pos = frame.to_h(g.accessor(a["POSITION"]))
    nrm = frame.dir_to_h(g.accessor(a["NORMAL"]))
    uvs = g.accessor(a["TEXCOORD_0"]).astype(np.float64)
    if "TANGENT" in a:
        tan = g.accessor(a["TANGENT"]).astype(np.float64)
        tan = np.concatenate([frame.dir_to_h(tan[:, :3]), tan[:, 3:]], 1)
    else:
        tan = np.zeros((len(pos), 4))
    tris = g.accessor(prim["indices"]).reshape(-1, 3).astype(np.int64)
    radius = float(fit.get("eye_params", {}).get("optics", {}).get("sclera_radius", optics.ANATOMICAL.sclera_radius))
    eyes = [{"side": e["side"], "center": frame.to_h(e["center"]), "rotation": frame.rot_to_h(e["rotation"]),
             "radius": radius} for e in eyes_glb]
    eyes.sort(key=lambda e: 0 if e["side"] == "right" else 1)
    return Subject(folder, frame, pos, nrm, uvs, tan, tris, eyes, fit, g)


# ---- geometry helpers -------------------------------------------------------------

def normalize(v, axis=-1):
    v = np.asarray(v, np.float64)
    return v / np.maximum(np.linalg.norm(v, axis=axis, keepdims=True), 1e-12)


def vertex_normals(pos: np.ndarray, tris: np.ndarray) -> np.ndarray:
    fn = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    n = np.zeros_like(pos)
    for c in range(3):
        np.add.at(n, tris[:, c], fn)
    return normalize(n)


def edges(tris: np.ndarray) -> np.ndarray:
    """Unique undirected edges (E, 2)."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e.sort(1)
    return np.unique(e, axis=0)


def boundary_edges(tris: np.ndarray) -> np.ndarray:
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    key = np.sort(e, 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return e[cnt[inv.reshape(-1)] == 1]


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def rotation(axis, angle) -> np.ndarray:
    """Rodrigues: (3,) axis, scalar or (N,) angles -> (3, 3) or (N, 3, 3)."""
    a = normalize(axis)
    ang = np.asarray(angle, np.float64)
    c, s = np.cos(ang)[..., None, None], np.sin(ang)[..., None, None]
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(a, a)


def quat_from_matrix(m: np.ndarray) -> np.ndarray:
    """(3, 3) rotation -> (x, y, z, w)."""
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def resample_polyline(pts: np.ndarray, n: int, closed: bool = False) -> np.ndarray:
    """n points evenly spaced by arc length."""
    p = np.asarray(pts, np.float64)
    if closed:
        p = np.vstack([p, p[:1]])
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n, endpoint=not closed)
    return np.stack([np.interp(t, s, p[:, c]) for c in range(p.shape[1])], 1)


def closest_on_segments(points: np.ndarray, poly: np.ndarray):
    """Distance from each point to a polyline (open), and the arc parameter
    in [0, 1] of the closest point."""
    a, b = poly[:-1], poly[1:]
    ab = b - a
    L = np.maximum((ab ** 2).sum(-1), 1e-18)
    ap = points[:, None, :] - a[None]
    t = np.clip((ap * ab[None]).sum(-1) / L, 0, 1)
    q = a[None] + t[..., None] * ab[None]
    d = np.linalg.norm(points[:, None, :] - q, axis=-1)
    j = np.argmin(d, 1)
    seg = np.linalg.norm(ab, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    s = (cum[j] + t[np.arange(len(points)), j] * seg[j]) / max(cum[-1], 1e-12)
    return d[np.arange(len(points)), j], s
