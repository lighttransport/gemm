"""Meshes carried over from the fitted head export (head_eyes.glb): the
eyeballs (rigid on the eye joints) and the eye-edge meshes (tearlines,
caruncles, occlusion shells), which follow the lids through a surface wrap:
each vertex is bound to its closest template triangle and inherits that
triangle's skin weights and blendshape deltas barycentrically."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from ..eye.glb import GLB
from .bake import _closest_on_triangles
from .common import Frame, normalize


@dataclass
class Carried:
    name: str
    positions: np.ndarray          # H frame
    normals: np.ndarray
    uv: np.ndarray
    tris: np.ndarray
    material: dict                 # glTF material (textures referenced by source index)
    joint: str | None = None       # rigid binding
    wrap: tuple | None = None      # (template tri ids, bary) wrap binding
    extra: dict = field(default_factory=dict)


def _node_matrix(n: dict) -> np.ndarray:
    from .common import rotation
    M = np.eye(4)
    if "matrix" in n:
        return np.asarray(n["matrix"]).reshape(4, 4).T
    t = np.asarray(n.get("translation", [0, 0, 0]), np.float64)
    q = np.asarray(n.get("rotation", [0, 0, 0, 1]), np.float64)
    s = np.asarray(n.get("scale", [1, 1, 1]), np.float64)
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M[:3, :3] = R * s
    M[:3, 3] = t
    return M


def collect(glb: GLB, frame: Frame) -> list[Carried]:
    doc = glb.doc
    parent = {}
    for i, n in enumerate(doc["nodes"]):
        for c in n.get("children", []):
            parent[c] = i

    def world(i):
        M = _node_matrix(doc["nodes"][i])
        while i in parent:
            i = parent[i]
            M = _node_matrix(doc["nodes"][i]) @ M
        return M

    out = []
    for i, n in enumerate(doc["nodes"]):
        if "mesh" not in n:
            continue
        name = n.get("name", f"node{i}")
        if name in ("head", "lid_lining"):
            continue                                   # replaced by the template / its lid strips
        mesh = doc["meshes"][n["mesh"]]
        prim = mesh["primitives"][0]
        a = prim["attributes"]
        P = glb.accessor(a["POSITION"]).astype(np.float64)
        N = glb.accessor(a["NORMAL"]).astype(np.float64) if "NORMAL" in a else np.zeros_like(P)
        UV = glb.accessor(a["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in a else np.zeros((len(P), 2))
        T = glb.accessor(prim["indices"]).reshape(-1, 3).astype(np.int64)
        M = world(i)
        Pw = P @ M[:3, :3].T + M[:3, 3]
        Nw = normalize(N @ np.linalg.inv(M[:3, :3]))
        c = Carried(name, frame.to_h(Pw), frame.dir_to_h(Nw), UV, T,
                    json.loads(json.dumps(doc["materials"][prim.get("material", 0)])))
        side = "R" if "right" in name else "L"
        if name.endswith("_shell") or name.endswith("_iris"):
            c.joint = f"eye_{side}"
        out.append(c)
    return out


def bind_wrap(points: np.ndarray, tmpl_pos: np.ndarray, tris: np.ndarray, k: int = 12):
    """Closest template triangle (among those nearest by centroid) and barycentrics."""
    from scipy.spatial import cKDTree
    cen = tmpl_pos[tris].mean(1)
    tree = cKDTree(cen)
    _, cand = tree.query(points, k=k)
    best_d = np.full(len(points), np.inf)
    best_t = np.zeros(len(points), np.int64)
    best_b = np.zeros((len(points), 3))
    for c in range(k):
        t = tris[cand[:, c]]
        q, b = _closest_on_triangles(points, tmpl_pos[t[:, 0]], tmpl_pos[t[:, 1]], tmpl_pos[t[:, 2]])
        d = np.linalg.norm(q - points, axis=1)
        m = d < best_d
        best_d[m], best_t[m], best_b[m] = d[m], cand[m, c], b[m]
    return best_t, best_b, best_d


def wrap_data(bind, tris, joints, weights, shapes: dict):
    """Per attached vertex: joint weights (on the template's joint set) and
    shape deltas, interpolated from the bound triangle."""
    t, b = bind
    v = tris[t]                                           # (n, 3) template vertex ids
    # the template's joint columns are the same for every vertex (root, neck, head, jaw)
    W = (b[:, :, None] * weights[v]).sum(1)
    J = joints[v[:, 0]]
    d = {k: (b[:, :, None] * s[v]).sum(1) for k, s in shapes.items()}
    return J, W, d
