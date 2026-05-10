"""cumesh shim for the ROCm runner.

Implements the subset of `cumesh` that `o_voxel.postprocess.to_glb` and
`Trellis2TexturingPipeline.postprocess_mesh` need:

  - cumesh.CuMesh: init/read, num_vertices, num_faces, fill_holes, simplify,
    remove_duplicate_faces, repair_non_manifold_edges,
    remove_small_connected_components, unify_face_orientations,
    compute_vertex_normals, read_vertex_normals, uv_unwrap.
  - cumesh.cuBVH: unsigned_distance(points, return_uvw=True).

All heavy work lives in C++:
  - common/libclosest_point_bvh.so  (lightrt SBVH closest-point)
  - common/libmesh_ops.so           (mesh edits + xatlas UV unwrap)

Python decimation is delegated to `fast_simplification` (vendored C++ via
pip wheel, no pure-Python algorithm).

Branch 2 of o_voxel.postprocess.to_glb (remesh=True via
cumesh.remeshing.remesh_narrow_band_dc) is NOT implemented — narrow-band
dual contouring would need its own port. Use remesh=False.
"""

import ctypes
import json
import os
import struct

import numpy as np
import torch


# ---------------------------------------------------------------------------
# safetensors I/O for mesh exchange (matches common/mesh_io.h schema).
#
#   "vertices":       F32 [V, 3]   required
#   "faces":          I32 [F, 3]   required
#   "vertex_normals": F32 [V, 3]   optional
#   "uvs":            F32 [V, 2]   optional
#   "vmap":           I32 [V]      optional
#
# Pure stdlib (struct + json + numpy) so the Python and C paths share the
# same file format with no extra deps.
# ---------------------------------------------------------------------------

_ST_DTYPE = {
    'F32': np.float32, 'F16': np.float16, 'F64': np.float64,
    'I8':  np.int8,    'U8':  np.uint8,
    'I16': np.int16,   'I32': np.int32,   'I64': np.int64,
    'BOOL': np.bool_,
}
_ST_DTYPE_REV = {v: k for k, v in _ST_DTYPE.items()}


def load_mesh_safetensors(path):
    """Load a mesh-shaped safetensors file. Returns dict of name -> ndarray."""
    with open(path, 'rb') as fp:
        n = struct.unpack('<Q', fp.read(8))[0]
        hdr = json.loads(fp.read(n))
        body = fp.read()
    out = {}
    for k, meta in hdr.items():
        if k == '__metadata__':
            continue
        dt = _ST_DTYPE[meta['dtype']]
        o0, o1 = meta['data_offsets']
        arr = np.frombuffer(body[o0:o1], dtype=dt)
        if meta.get('shape'):
            arr = arr.reshape(meta['shape'])
        out[k] = arr
    if 'vertices' not in out or 'faces' not in out:
        raise ValueError(f"{path}: missing 'vertices' or 'faces'")
    return out


def save_mesh_safetensors(path, vertices, faces, *,
                          vertex_normals=None, uvs=None, vmap=None,
                          extras=None):
    """Save a mesh as safetensors. Accepts torch tensors or ndarrays.

    extras: optional dict[name -> ndarray] of additional tensors to append.
    """
    def _to_np(x, dtype, ndim_check=None):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        x = np.ascontiguousarray(np.asarray(x), dtype=dtype)
        if ndim_check is not None and x.ndim != ndim_check:
            raise ValueError(f'expected ndim={ndim_check}, got {x.shape}')
        return x

    v = _to_np(vertices,       np.float32, 2)
    f = _to_np(faces,          np.int32,   2)
    n = _to_np(vertex_normals, np.float32, 2)
    u = _to_np(uvs,            np.float32, 2)
    m = _to_np(vmap,           np.int32,   1)

    items = [('vertices', v), ('faces', f)]
    if n is not None: items.append(('vertex_normals', n))
    if u is not None: items.append(('uvs', u))
    if m is not None: items.append(('vmap', m))
    if extras:
        for k, arr in extras.items():
            arr = np.ascontiguousarray(np.asarray(arr))
            items.append((k, arr))

    hdr = {}
    cur = 0
    blobs = []
    for name, arr in items:
        nbytes = arr.nbytes
        dt_str = _ST_DTYPE_REV.get(arr.dtype.type)
        if dt_str is None:
            raise ValueError(f'unsupported dtype {arr.dtype} for {name}')
        hdr[name] = {
            'dtype': dt_str,
            'shape': list(arr.shape),
            'data_offsets': [cur, cur + nbytes],
        }
        cur += nbytes
        blobs.append(arr.tobytes())
    hdr_b = json.dumps(hdr, separators=(',', ':')).encode()
    pad = (8 - (len(hdr_b) % 8)) % 8
    hdr_b = hdr_b + b' ' * pad
    with open(path, 'wb') as fp:
        fp.write(struct.pack('<Q', len(hdr_b)))
        fp.write(hdr_b)
        for b in blobs:
            fp.write(b)


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


def _common_dir():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, '..', '..', '..', 'common'))


# ---------------------------------------------------------------------------
# libmesh_ops.so wrapper
# ---------------------------------------------------------------------------


class _MeshOpsLib:
    _LIB = None

    @classmethod
    def get(cls):
        if cls._LIB is not None:
            return cls._LIB
        so = os.path.join(_common_dir(), 'libmesh_ops.so')
        if not os.path.isfile(so):
            raise RuntimeError(
                f'libmesh_ops.so not found at {so}; build with '
                f'`make -C common -f Makefile.mesh_ops`')
        lib = ctypes.CDLL(so)

        # mesh handle
        lib.tr2_mesh_create.restype = ctypes.c_void_p
        lib.tr2_mesh_destroy.argtypes = [ctypes.c_void_p]
        lib.tr2_mesh_set.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_uint32,
            ctypes.c_void_p, ctypes.c_uint32]
        lib.tr2_mesh_num_v.argtypes = [ctypes.c_void_p]
        lib.tr2_mesh_num_v.restype = ctypes.c_uint32
        lib.tr2_mesh_num_f.argtypes = [ctypes.c_void_p]
        lib.tr2_mesh_num_f.restype = ctypes.c_uint32
        lib.tr2_mesh_get_v.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.tr2_mesh_get_f.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        # ops
        lib.tr2_mesh_compute_vertex_normals.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.tr2_mesh_remove_duplicate_faces.argtypes = [ctypes.c_void_p]
        lib.tr2_mesh_remove_small_components.argtypes = [ctypes.c_void_p, ctypes.c_float]
        lib.tr2_mesh_unify_orientations.argtypes = [ctypes.c_void_p]
        lib.tr2_mesh_fill_holes.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        # uv unwrap
        lib.tr2_unwrap_create.restype = ctypes.c_void_p
        lib.tr2_unwrap_destroy.argtypes = [ctypes.c_void_p]
        lib.tr2_mesh_uv_unwrap.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.tr2_mesh_uv_unwrap.restype = ctypes.c_int
        lib.tr2_unwrap_num_v.argtypes = [ctypes.c_void_p]
        lib.tr2_unwrap_num_v.restype = ctypes.c_uint32
        lib.tr2_unwrap_num_f.argtypes = [ctypes.c_void_p]
        lib.tr2_unwrap_num_f.restype = ctypes.c_uint32
        lib.tr2_unwrap_get_v.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.tr2_unwrap_get_f.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.tr2_unwrap_get_uv.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.tr2_unwrap_get_vmap.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cls._LIB = lib
        return lib


class CuMesh:
    def __init__(self):
        self._lib = _MeshOpsLib.get()
        self._handle = ctypes.c_void_p(self._lib.tr2_mesh_create())
        self._vnormals = None  # numpy [V,3] float32

    def __del__(self):
        try:
            if getattr(self, '_handle', None) and self._handle.value:
                self._lib.tr2_mesh_destroy(self._handle)
                self._handle = None
        except Exception:
            pass

    # ---- init / read ------------------------------------------------------
    def init(self, vertices, faces):
        v = np.ascontiguousarray(_to_np_v(vertices), dtype=np.float32)
        f = np.ascontiguousarray(_to_np_f(faces).astype(np.int32, copy=False))
        nv = int(v.shape[0])
        nf = int(f.shape[0])
        self._lib.tr2_mesh_set(self._handle,
                               v.ctypes.data, nv,
                               f.ctypes.data, nf)
        self._vnormals = None

    def _read_v(self):
        nv = int(self._lib.tr2_mesh_num_v(self._handle))
        out = np.empty((nv, 3), dtype=np.float32)
        if nv:
            self._lib.tr2_mesh_get_v(self._handle, out.ctypes.data)
        return out

    def _read_f(self):
        nf = int(self._lib.tr2_mesh_num_f(self._handle))
        out = np.empty((nf, 3), dtype=np.int32)
        if nf:
            self._lib.tr2_mesh_get_f(self._handle, out.ctypes.data)
        return out

    def read(self):
        v = self._read_v()
        f = self._read_f()
        return _maybe_cuda(torch.from_numpy(v)), _maybe_cuda(torch.from_numpy(f))

    @property
    def num_vertices(self):
        return int(self._lib.tr2_mesh_num_v(self._handle))

    @property
    def num_faces(self):
        return int(self._lib.tr2_mesh_num_f(self._handle))

    # ---- editing ----------------------------------------------------------
    def fill_holes(self, max_hole_perimeter=None):
        # max_hole_perimeter is treated as max-edge-count if int-ish, else 64.
        n = 0
        if max_hole_perimeter is not None:
            try:
                n = max(0, int(max_hole_perimeter))
            except Exception:
                n = 0
        self._lib.tr2_mesh_fill_holes(self._handle, ctypes.c_uint32(n))
        self._vnormals = None

    def simplify(self, target, verbose=False, options=None):
        # Open3D quadric-error decimation — matches upstream cumesh behavior
        # (which is what o_voxel.postprocess.to_glb expects).
        import open3d as o3d
        if self.num_faces <= int(target):
            return
        v = self._read_v().astype(np.float64, copy=False)
        f = self._read_f().astype(np.int32, copy=False)
        tm = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(v),
            o3d.utility.Vector3iVector(f),
        )
        tm = tm.simplify_quadric_decimation(target_number_of_triangles=int(target))
        v2 = np.asarray(tm.vertices, dtype=np.float32)
        f2 = np.asarray(tm.triangles, dtype=np.int32)
        v2 = np.ascontiguousarray(v2)
        f2 = np.ascontiguousarray(f2)
        self._lib.tr2_mesh_set(self._handle,
                               v2.ctypes.data, int(v2.shape[0]),
                               f2.ctypes.data, int(f2.shape[0]))
        self._vnormals = None

    def remove_duplicate_faces(self):
        self._lib.tr2_mesh_remove_duplicate_faces(self._handle)
        self._vnormals = None

    def repair_non_manifold_edges(self):
        # No-op: xatlas tolerates non-manifold inputs.
        pass

    def remove_small_connected_components(self, area_threshold):
        self._lib.tr2_mesh_remove_small_components(
            self._handle, ctypes.c_float(float(area_threshold)))
        self._vnormals = None

    def unify_face_orientations(self):
        self._lib.tr2_mesh_unify_orientations(self._handle)
        self._vnormals = None

    # ---- normals ----------------------------------------------------------
    def compute_vertex_normals(self):
        nv = self.num_vertices
        out = np.empty((nv, 3), dtype=np.float32)
        if nv:
            self._lib.tr2_mesh_compute_vertex_normals(
                self._handle, out.ctypes.data)
        self._vnormals = out

    def read_vertex_normals(self):
        if self._vnormals is None:
            self.compute_vertex_normals()
        return _maybe_cuda(torch.from_numpy(np.ascontiguousarray(self._vnormals)))

    # ---- safetensors I/O (matches common/mesh_io.h schema) ----------------
    def load_safetensors(self, path):
        d = load_mesh_safetensors(path)
        self.init(d['vertices'], d['faces'])
        if 'vertex_normals' in d:
            self._vnormals = np.ascontiguousarray(
                d['vertex_normals'], dtype=np.float32)
        return d  # caller may also want uvs/vmap

    def save_safetensors(self, path, *, with_normals=False, uvs=None, vmap=None):
        v = self._read_v()
        f = self._read_f()
        n = None
        if with_normals:
            if self._vnormals is None:
                self.compute_vertex_normals()
            n = self._vnormals
        save_mesh_safetensors(path, v, f,
                              vertex_normals=n, uvs=uvs, vmap=vmap)

    # ---- UV unwrap (xatlas via libmesh_ops.so) ----------------------------
    def uv_unwrap(self, return_vmaps=False, compute_charts_kwargs=None,
                  verbose=False):
        unwrap_h = ctypes.c_void_p(self._lib.tr2_unwrap_create())
        try:
            rc = self._lib.tr2_mesh_uv_unwrap(self._handle, unwrap_h)
            if rc != 0:
                raise RuntimeError('tr2_mesh_uv_unwrap failed')
            nv = int(self._lib.tr2_unwrap_num_v(unwrap_h))
            nf = int(self._lib.tr2_unwrap_num_f(unwrap_h))
            v_out = np.empty((nv, 3), dtype=np.float32)
            f_out = np.empty((nf, 3), dtype=np.int32)
            uv_out = np.empty((nv, 2), dtype=np.float32)
            vmap_out = np.empty((nv,), dtype=np.int32)
            if nv:
                self._lib.tr2_unwrap_get_v(unwrap_h, v_out.ctypes.data)
                self._lib.tr2_unwrap_get_uv(unwrap_h, uv_out.ctypes.data)
                self._lib.tr2_unwrap_get_vmap(unwrap_h, vmap_out.ctypes.data)
            if nf:
                self._lib.tr2_unwrap_get_f(unwrap_h, f_out.ctypes.data)
        finally:
            self._lib.tr2_unwrap_destroy(unwrap_h)

        verts_out = _maybe_cuda(torch.from_numpy(v_out))
        faces_out = _maybe_cuda(torch.from_numpy(f_out))
        uvs_out = _maybe_cuda(torch.from_numpy(uv_out))
        if return_vmaps:
            vmap_t = _maybe_cuda(torch.from_numpy(vmap_out.astype(np.int64)))
            return verts_out, faces_out, uvs_out, vmap_t
        return verts_out, faces_out, uvs_out


# ---------------------------------------------------------------------------
# libclosest_point_bvh.so wrapper (lightrt SBVH)
# ---------------------------------------------------------------------------


class _LightrtBVH:
    """ctypes wrapper around common/libclosest_point_bvh.so (lightrt SBVH)."""
    _LIB = None

    @classmethod
    def _load(cls):
        if cls._LIB is not None:
            return cls._LIB
        so = os.path.join(_common_dir(), 'libclosest_point_bvh.so')
        if not os.path.isfile(so):
            raise RuntimeError(
                f'libclosest_point_bvh.so not found at {so}; build with '
                f'`make -C common -f Makefile.closest_point_bvh`')
        lib = ctypes.CDLL(so)
        lib.tr2_cpbvh_create.restype = ctypes.c_void_p
        lib.tr2_cpbvh_create.argtypes = [
            ctypes.c_void_p, ctypes.c_uint32,
            ctypes.c_void_p, ctypes.c_uint32]
        lib.tr2_cpbvh_destroy.argtypes = [ctypes.c_void_p]
        lib.tr2_cpbvh_query.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_uint32,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_uint32]
        cls._LIB = lib
        return lib

    def __init__(self, v, f):
        lib = self._load()
        v32 = np.ascontiguousarray(v, dtype=np.float32)
        f32 = np.ascontiguousarray(f, dtype=np.int32)
        self._v_ref = v32
        self._f_ref = f32
        self._handle = ctypes.c_void_p(lib.tr2_cpbvh_create(
            v32.ctypes.data, v32.shape[0],
            f32.ctypes.data, f32.shape[0]))
        if not self._handle.value:
            raise RuntimeError('tr2_cpbvh_create returned NULL')
        self._lib = lib

    def __del__(self):
        try:
            if getattr(self, '_handle', None) and self._handle.value:
                self._lib.tr2_cpbvh_destroy(self._handle)
                self._handle = None
        except Exception:
            pass

    def query(self, pts):
        N = pts.shape[0]
        face = np.empty(N, dtype=np.uint32)
        dist = np.empty(N, dtype=np.float32)
        uvw  = np.empty((N, 3), dtype=np.float32)
        clos = np.empty((N, 3), dtype=np.float32)
        self._lib.tr2_cpbvh_query(
            self._handle, pts.ctypes.data, N,
            face.ctypes.data, dist.ctypes.data,
            uvw.ctypes.data, clos.ctypes.data, 0)
        return face, dist, uvw, clos


class cuBVH:
    """Closest-point queries on the input mesh, backed by lightrt SBVH.

    Used by o_voxel.postprocess.to_glb to map texel positions on the cleaned
    mesh back to the original (pre-simplification) mesh for accurate attr
    sampling. No fallback — libclosest_point_bvh.so is required.
    """

    def __init__(self, vertices, faces):
        v = _to_np_v(vertices)
        f = _to_np_f(faces).astype(np.int32, copy=False)
        self._impl = _LightrtBVH(v, f)

    def unsigned_distance(self, points, return_uvw=False):
        device = points.device if isinstance(points, torch.Tensor) else 'cpu'
        if isinstance(points, torch.Tensor):
            pts = points.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            pts = np.asarray(points, dtype=np.float32)
        pts = np.ascontiguousarray(pts)

        face_id, dist, uvw, _closest = self._impl.query(pts)
        face_id = face_id.astype(np.int64)

        dist_t = torch.from_numpy(np.ascontiguousarray(dist)).to(device)
        face_t = torch.from_numpy(np.ascontiguousarray(face_id)).to(device)
        if not return_uvw:
            return dist_t, face_t
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
