"""Host-side readers for the raw meshes and glTF files written by the UE scripts.

Binary format (ue_scripts/smoke_03_geometry.py, gen_batch.py): b'VHMH', uint32 nverts, uint32 ntris, float32 xyz*nverts,
int32 3*ntris. Units are Unreal centimetres in the component's local frame.
"""
import json
import struct
from pathlib import Path

import numpy as np

MAGIC = b'VHMH'
GLTF_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
GLTF_COUNT = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4}


def read_mesh_bin(path):
    data = Path(path).read_bytes()
    if data[:4] != MAGIC:
        raise ValueError(f'{path}: bad magic')
    nv, nt = struct.unpack_from('<II', data, 4)
    off = 12
    verts = np.frombuffer(data, np.float32, 3 * nv, off).reshape(nv, 3).copy(); off += 12 * nv
    tris = np.frombuffer(data, np.int32, 3 * nt, off).reshape(nt, 3).copy()
    return verts, tris


def connected_components(nv, tris):
    """Label vertices by mesh connectivity (union-find); returns labels (nv,) and sizes."""
    parent = np.arange(nv)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b, c in tris:
        ra, rb, rc = find(a), find(b), find(c)
        parent[rb] = ra; parent[rc] = ra
    roots = np.array([find(i) for i in range(nv)])
    _, labels, sizes = np.unique(roots, return_inverse=True, return_counts=True)
    return labels, sizes


def read_gltf(path):
    """Return dict of primitives: positions (n,3), uvs (n,2) (glTF convention, V down), indices (m,3), per mesh/primitive."""
    path = Path(path)
    g = json.loads(path.read_text())
    buffers = []
    for b in g['buffers']:
        uri = b.get('uri')
        if uri and not uri.startswith('data:'):
            buffers.append((path.parent / uri).read_bytes())
        elif uri:
            import base64
            buffers.append(base64.b64decode(uri.split(',', 1)[1]))
        else:
            raise ValueError('GLB not supported here')

    def accessor(i):
        a = g['accessors'][i]; bv = g['bufferViews'][a['bufferView']]
        dtype = GLTF_COMPONENT[a['componentType']]; n = GLTF_COUNT[a['type']]
        buf = buffers[bv['buffer']]
        off = bv.get('byteOffset', 0) + a.get('byteOffset', 0)
        stride = bv.get('byteStride', 0)
        count = a['count']
        if stride and stride != n * np.dtype(dtype).itemsize:
            raw = np.frombuffer(buf, np.uint8, stride * count, off).reshape(count, stride)[:, :n * np.dtype(dtype).itemsize]
            return np.frombuffer(raw.tobytes(), dtype).reshape(count, n)
        return np.frombuffer(buf, dtype, count * n, off).reshape(count, n).copy()
    prims = []
    for mi, mesh in enumerate(g.get('meshes', [])):
        for pi, p in enumerate(mesh['primitives']):
            attrs = p['attributes']
            rec = dict(mesh=mesh.get('name', str(mi)), primitive=pi, positions=accessor(attrs['POSITION']).astype(np.float32))
            if 'TEXCOORD_0' in attrs:
                rec['uvs'] = accessor(attrs['TEXCOORD_0']).astype(np.float32)
            if 'NORMAL' in attrs:
                rec['normals'] = accessor(attrs['NORMAL']).astype(np.float32)
            if 'indices' in p:
                rec['indices'] = accessor(p['indices']).reshape(-1, 3).astype(np.int64)
            rec['material'] = p.get('material')
            prims.append(rec)
    return dict(primitives=prims, materials=[m.get('name') for m in g.get('materials', [])], nodes=g.get('nodes', []))


def gltf_to_ue(positions):
    """glTF exporter writes metres, right-handed Y-up; UE is cm, Z-up, left-handed: (x, y, z)_gltf -> (x, -z... ) handled by caller.

    The exact axis mapping is verified in smoke 03 by comparing against the VHMH binary of the same mesh; this helper
    only rescales to cm. Keep it explicit so the mapping lives in one place once measured.
    """
    return positions * 100.0
