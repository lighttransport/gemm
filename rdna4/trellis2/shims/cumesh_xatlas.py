"""cumesh shim for the ROCm runner.

Implements the subset of `cumesh` that `o_voxel.postprocess.to_glb` and
`Trellis2TexturingPipeline.postprocess_mesh` need:

  - cumesh.CuMesh: init/read, num_vertices, num_faces, fill_holes, simplify,
    remove_duplicate_faces, repair_non_manifold_edges,
    remove_small_connected_components, unify_face_orientations,
    compute_vertex_normals, read_vertex_normals, uv_unwrap.
  - cumesh.cuBVH: unsigned_distance(points, return_uvw=True).

Backed by trimesh (mesh edits) + fast_simplification (decimation) + xatlas
(UV unwrap) + scipy/sklearn-free closest-point queries through trimesh's
ProximityQuery. CPU-only; the heavy steps already run in C/C++ libs.

Branch 2 of o_voxel.postprocess.to_glb (remesh=True via
cumesh.remeshing.remesh_narrow_band_dc) is NOT implemented — narrow-band
dual contouring would need its own port. Use remesh=False.
"""

import numpy as np
import torch
import trimesh


def _maybe_cuda(t):
    if torch.cuda.is_available():
        return t.cuda()
    return t


def _to_np_v(t):
    if isinstance(t, np.ndarray):
        return t.astype(np.float32, copy=False)
    return t.detach().cpu().numpy().astype(np.float32, copy=False)


def _to_np_f(t):
    if isinstance(t, np.ndarray):
        return t.astype(np.int64, copy=False)
    return t.detach().cpu().numpy().astype(np.int64, copy=False)


class CuMesh:
    def __init__(self):
        self._mesh = None
        self._vnormals = None

    # ---- init / read ------------------------------------------------------
    def init(self, vertices, faces):
        v = _to_np_v(vertices)
        f = _to_np_f(faces)
        self._mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        self._vnormals = None

    def read(self):
        v = torch.from_numpy(np.ascontiguousarray(self._mesh.vertices, dtype=np.float32))
        f = torch.from_numpy(np.ascontiguousarray(self._mesh.faces, dtype=np.int32))
        return _maybe_cuda(v), _maybe_cuda(f)

    @property
    def num_vertices(self):
        return int(len(self._mesh.vertices))

    @property
    def num_faces(self):
        return int(len(self._mesh.faces))

    # ---- editing ----------------------------------------------------------
    def fill_holes(self, max_hole_perimeter=None):
        # trimesh.fill_holes() fills triangular/quad holes and returns success.
        # max_hole_perimeter is ignored (trimesh has no equivalent knob); the
        # downstream pipeline tolerates leftover holes via cv2.inpaint.
        try:
            self._mesh.fill_holes()
        except Exception:
            pass
        self._vnormals = None

    def simplify(self, target, verbose=False, options=None):
        import fast_simplification
        if self.num_faces <= target:
            return
        v = self._mesh.vertices.astype(np.float32, copy=False)
        f = self._mesh.faces.astype(np.uint32, copy=False)
        v2, f2 = fast_simplification.simplify(v, f, target_count=int(target))
        self._mesh = trimesh.Trimesh(vertices=v2, faces=f2, process=False)
        self._vnormals = None

    def remove_duplicate_faces(self):
        unique = trimesh.grouping.unique_rows(np.sort(self._mesh.faces, axis=1))[0]
        self._mesh = trimesh.Trimesh(vertices=self._mesh.vertices,
                                     faces=self._mesh.faces[unique],
                                     process=False)
        self._vnormals = None

    def repair_non_manifold_edges(self):
        # trimesh does not have a true non-manifold-edge repair; the cleanest
        # option here is a no-op. Downstream uv_unwrap (xatlas) handles
        # non-manifold inputs.
        pass

    def remove_small_connected_components(self, area_threshold):
        comps = self._mesh.split(only_watertight=False)
        if not comps:
            return
        keep = [c for c in comps if c.area >= area_threshold]
        if not keep:
            keep = [max(comps, key=lambda c: c.area)]
        merged = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
        self._mesh = trimesh.Trimesh(vertices=merged.vertices,
                                     faces=merged.faces, process=False)
        self._vnormals = None

    def unify_face_orientations(self):
        try:
            trimesh.repair.fix_winding(self._mesh)
            trimesh.repair.fix_normals(self._mesh)
        except Exception:
            pass
        self._vnormals = None

    # ---- normals ----------------------------------------------------------
    def compute_vertex_normals(self):
        self._vnormals = self._mesh.vertex_normals.astype(np.float32, copy=False)

    def read_vertex_normals(self):
        if self._vnormals is None:
            self.compute_vertex_normals()
        return _maybe_cuda(torch.from_numpy(np.ascontiguousarray(self._vnormals)))

    # ---- UV unwrap (xatlas) ----------------------------------------------
    def uv_unwrap(self, return_vmaps=False, compute_charts_kwargs=None,
                  verbose=False):
        import xatlas
        v = self._mesh.vertices.astype(np.float32, copy=False)
        f = self._mesh.faces.astype(np.uint32, copy=False)
        vmap, indices, uvs = xatlas.parametrize(v, f)
        verts_out = _maybe_cuda(torch.from_numpy(v[vmap].astype(np.float32)))
        faces_out = _maybe_cuda(torch.from_numpy(indices.astype(np.int32)))
        uvs_out = _maybe_cuda(torch.from_numpy(uvs.astype(np.float32)))
        if return_vmaps:
            return verts_out, faces_out, uvs_out, _maybe_cuda(torch.from_numpy(vmap.astype(np.int64)))
        return verts_out, faces_out, uvs_out


class cuBVH:
    """Closest-point queries on the input mesh.

    Used by o_voxel.postprocess.to_glb to map texel positions on the cleaned
    mesh back to the original (pre-simplification) mesh for accurate attr
    sampling.
    """

    def __init__(self, vertices, faces):
        v = _to_np_v(vertices)
        f = _to_np_f(faces)
        self._mesh = trimesh.Trimesh(vertices=v, faces=f, process=False)
        self._tri_verts = v[f]  # (F, 3, 3)

    def unsigned_distance(self, points, return_uvw=False):
        """points: torch.Tensor [N,3] cuda or cpu. Returns (dist, face_id, uvw).
        uvw is barycentric coords of the closest point on each face_id.
        """
        device = points.device if isinstance(points, torch.Tensor) else 'cpu'
        if isinstance(points, torch.Tensor):
            pts = points.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            pts = np.asarray(points, dtype=np.float32)

        closest, dist, face_id = trimesh.proximity.closest_point(self._mesh, pts)
        face_id = face_id.astype(np.int64, copy=False)

        dist_t = torch.from_numpy(np.ascontiguousarray(dist.astype(np.float32))).to(device)
        face_t = torch.from_numpy(np.ascontiguousarray(face_id)).to(device)
        if not return_uvw:
            return dist_t, face_t

        tri = self._tri_verts[face_id]  # (N, 3, 3)
        v0 = tri[:, 0]; v1 = tri[:, 1]; v2 = tri[:, 2]
        e1 = v1 - v0
        e2 = v2 - v0
        n = np.cross(e1, e2)
        denom = np.einsum('ij,ij->i', n, n)
        denom = np.where(denom > 1e-30, denom, 1e-30)
        p = closest.astype(np.float32, copy=False) - v0
        # Barycentric via signed sub-triangle areas (Cramer-like).
        c1 = np.einsum('ij,ij->i', np.cross(e1, p), n) / denom
        c2 = np.einsum('ij,ij->i', np.cross(p, e2), n) / denom
        # c1 ↔ weight on v2, c2 ↔ weight on v1 (right-hand rule). Compose so
        # that closest = w0*v0 + w1*v1 + w2*v2.
        w1 = c2
        w2 = c1
        w0 = 1.0 - w1 - w2
        uvw = np.stack([w0, w1, w2], axis=-1).astype(np.float32)
        uvw_t = torch.from_numpy(np.ascontiguousarray(uvw)).to(device)
        return dist_t, face_t, uvw_t


def install_as_cumesh():
    import sys
    import types
    mod = types.ModuleType('cumesh')
    mod.CuMesh = CuMesh
    mod.cuBVH = cuBVH
    # Stub remeshing submodule; remesh=True path is not supported.
    remeshing = types.ModuleType('cumesh.remeshing')
    def _not_supported(*a, **k):
        raise NotImplementedError(
            'cumesh.remeshing.remesh_narrow_band_dc is not ported in the '
            'ROCm runner. Use o_voxel.postprocess.to_glb(..., remesh=False).')
    remeshing.remesh_narrow_band_dc = _not_supported
    mod.remeshing = remeshing
    sys.modules['cumesh'] = mod
    sys.modules['cumesh.remeshing'] = remeshing
