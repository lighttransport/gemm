"""Render/export meshes from the welded template: split by material and UV.

The rig works on welded vertices (one per template vertex): skin weights and
blendshape deltas are per welded vertex. Exporters need one vertex per
(vertex, uv, material); `unweld` builds those arrays plus the map back to
the welded vertex, so any per-vertex rig data can be gathered."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..eye.geometry import compute_tangents
from .common import normalize

MATERIALS = ("skin", "mouth")


@dataclass
class Part:
    name: str
    material: str
    vmap: np.ndarray          # (n,) welded vertex id per unwelded vertex
    uv: np.ndarray            # (n, 2)
    tris: np.ndarray          # (t, 3) unwelded indices
    normals: np.ndarray       # (n, 3) (from the material's own triangles)
    tangents: np.ndarray      # (n, 4)


def material_normals(pos: np.ndarray, tris: np.ndarray, n: int) -> np.ndarray:
    fn = np.cross(pos[tris[:, 1]] - pos[tris[:, 0]], pos[tris[:, 2]] - pos[tris[:, 0]])
    acc = np.zeros((n, 3))
    for c in range(3):
        np.add.at(acc, tris[:, c], fn)
    return normalize(acc)


def unweld(tmpl, pos: np.ndarray) -> list[Part]:
    parts = []
    for m, name in enumerate(MATERIALS):
        sel = tmpl.tri_mat == m
        tv = tmpl.tris[sel]
        tu = tmpl.tri_uv[sel]
        pairs = np.stack([tv.reshape(-1), tu.reshape(-1)], 1)
        uniq, inv = np.unique(pairs, axis=0, return_inverse=True)
        vmap, uvid = uniq[:, 0], uniq[:, 1]
        tris = inv.reshape(-1, 3)
        nrm_w = material_normals(pos, tv, len(pos))
        nrm = nrm_w[vmap]
        uv = tmpl.uv[uvid]
        tan = compute_tangents(pos[vmap], nrm, uv, tris).astype(np.float64)
        parts.append(Part(name, name, vmap, uv, tris, nrm, tan))
    return parts


def part_normals(part: Part, pos: np.ndarray) -> np.ndarray:
    """Normals of a part for deformed welded positions."""
    return material_normals(pos, part.vmap[part.tris], len(pos))[part.vmap]
