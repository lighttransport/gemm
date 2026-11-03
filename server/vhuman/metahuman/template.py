"""MetaHuman archetype head template: DNA-ordered vertices, triangles and per-corner UVs, built once from smoke outputs.

Inputs: the VHMH dump of the preview 'Face' component (head = vertices 0..24048 in DNA order) and the glTF export of the
same character's head skeletal mesh (UV-seam-split vertices in metres, already in the vhuman frame: x subject-left, y up,
z forward; verified: glTF (x,y,z) == UE (x,z,y)/100). Each glTF vertex is matched to its DNA vertex by position.
Output (derived geometry, stored outside the repo): MODELS_ROOT/binding_v1/mh_head_template.npz with
  vertices (24049,3) metres vhuman frame, triangles (F,3) DNA ids, triangle_uvs (F,3,2) texture coords with rows = v*H
  (UE and glTF UVs both have the origin at the top-left, matching the exported PNG rows), uv_vertex (24414,) gltf->dna.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import DATASET_ROOT, MODELS_ROOT
from .mesh_io import read_gltf, read_mesh_bin
from .samples import HEAD_VERTS, to_vhuman

DEFAULT_OUT = MODELS_ROOT / 'binding_v1' / 'mh_head_template.npz'


def build(face_bin, gltf, out=DEFAULT_OUT, tol_m=1e-5):
    verts, _ = read_mesh_bin(face_bin)
    head = to_vhuman(verts[:HEAD_VERTS])
    g = read_gltf(gltf)
    prim = max(g['primitives'], key=lambda p: len(p['positions']))         # the head is the largest primitive
    pos, uvs, idx = prim['positions'], prim['uvs'], prim['indices']
    d, dna = cKDTree(head).query(pos)
    if d.max() > tol_m:
        raise ValueError(f'glTF head vertices do not match the DNA head (max {d.max():.2e} m)')
    covered = np.unique(dna)
    if len(covered) != HEAD_VERTS:
        raise ValueError(f'glTF head covers {len(covered)} of {HEAD_VERTS} DNA vertices')
    tris = dna[idx]
    tri_uvs = uvs[idx].astype(np.float32)
    # orientation check: triangle normals should point outward (away from the head centre) for most faces
    a, b, c = (head[tris[:, i]] for i in range(3))
    n = np.cross(b - a, c - a); centre = head.mean(0)
    outward = float(((n * ((a + b + c) / 3 - centre)).sum(1) > 0).mean())
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, vertices=head.astype(np.float32), triangles=tris.astype(np.int32), triangle_uvs=tri_uvs,
                        uv_vertex=dna.astype(np.int32), outward_fraction=outward, source_face_bin=str(face_bin), source_gltf=str(gltf))
    return dict(out=str(out), vertices=len(head), triangles=len(tris), gltf_vertices=len(pos), outward_fraction=outward,
                uv_range=[tri_uvs.reshape(-1, 2).min(0).tolist(), tri_uvs.reshape(-1, 2).max(0).tolist()])


def load(path=DEFAULT_OUT):
    z = np.load(path)
    return {k: z[k] for k in ('vertices', 'triangles', 'triangle_uvs', 'uv_vertex')}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    smoke = DATASET_ROOT / 'smoke' / '03_geo'
    ap.add_argument('--face-bin', default=str(smoke / 'Face.vhmh.bin'))
    ap.add_argument('--gltf', default=str(smoke / 'Smoke03_Head.gltf'))
    ap.add_argument('--out', default=str(DEFAULT_OUT))
    a = ap.parse_args()
    print(build(a.face_bin, a.gltf, a.out))


if __name__ == '__main__':
    main()
