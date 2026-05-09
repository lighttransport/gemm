#!/usr/bin/env python3
"""Convert a mesh from OBJ or {vertices.npy, faces.npy} to mesh.safetensors
(schema from common/mesh_io.h: F32 vertices [V,3] + I32 faces [F,3]).

Usage:
  mesh_to_st.py --obj a.obj --out a.safetensors
  mesh_to_st.py --verts v.npy --faces f.npy --out a.safetensors
"""
import argparse, numpy as np
from safetensors.numpy import save_file

def from_obj(path):
    import trimesh
    m = trimesh.load(path, process=False, force='mesh')
    return np.asarray(m.vertices, dtype=np.float32), np.asarray(m.faces, dtype=np.int32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--obj')
    ap.add_argument('--verts')
    ap.add_argument('--faces')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if a.obj:
        v, f = from_obj(a.obj)
    else:
        v = np.load(a.verts).astype(np.float32, copy=False)
        f = np.load(a.faces).astype(np.int32, copy=False)
    if v.ndim != 2 or v.shape[1] != 3:
        raise SystemExit(f'vertices must be [V,3], got {v.shape}')
    if f.ndim != 2 or f.shape[1] != 3:
        raise SystemExit(f'faces must be [F,3], got {f.shape}')
    save_file({'vertices': np.ascontiguousarray(v),
               'faces': np.ascontiguousarray(f)}, a.out)
    print(f'wrote {a.out}: V={v.shape[0]} F={f.shape[0]}')

if __name__ == '__main__':
    main()
