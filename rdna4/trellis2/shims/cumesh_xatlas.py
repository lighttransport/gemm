"""Minimal cumesh shim backed by xatlas.

Only replaces the one call site in trellis2_texturing.py:
    _cumesh = cumesh.CuMesh(); _cumesh.init(verts, faces)
    verts_out, faces_out, uvs_out, vmap = _cumesh.uv_unwrap(return_vmaps=True)
"""

import numpy as np
import torch


class CuMesh:
    def __init__(self):
        self._verts = None
        self._faces = None

    def init(self, vertices, faces):
        import xatlas  # lazy: not required for --skip-dit paths
        self._xatlas = xatlas
        self._verts = vertices.detach().cpu().numpy().astype(np.float32)
        self._faces = faces.detach().cpu().numpy().astype(np.uint32)

    def uv_unwrap(self, return_vmaps=False):
        import xatlas
        vmap, indices, uvs = xatlas.parametrize(self._verts, self._faces)
        verts_out = torch.from_numpy(self._verts[vmap]).float()
        faces_out = torch.from_numpy(indices.astype(np.int32)).int()
        uvs_out = torch.from_numpy(uvs.astype(np.float32))
        if return_vmaps:
            return verts_out, faces_out, uvs_out, torch.from_numpy(vmap.astype(np.int64))
        return verts_out, faces_out, uvs_out


def install_as_cumesh():
    import sys
    import types
    mod = types.ModuleType('cumesh')
    mod.CuMesh = CuMesh
    sys.modules['cumesh'] = mod
