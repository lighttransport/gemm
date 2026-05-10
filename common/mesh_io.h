// Mesh <-> safetensors I/O helpers.
//
// File schema:
//   "vertices":       F32 [V, 3]   required
//   "faces":          I32 [F, 3]   required
//   "vertex_normals": F32 [V, 3]   optional
//   "uvs":            F32 [V, 2]   optional
//   "vmap":           I32 [V]      optional (xref to source mesh after unwrap)
//
// This is the on-disk contract between the trellis2 pipeline stages:
//   shapegen (C runner)   -> writes vertices + faces
//   mesh_proc (mesh_ops)  -> reads input mesh, writes cleaned mesh
//   texgen (C runner)     -> reads cleaned mesh + uvs/vmap

#pragma once

#include "mesh_ops.h"
#include "safetensors.h"
#include "safetensors_writer.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace trellis2 {

inline bool loadMesh(const char* path, Mesh& m) {
  st_context* ctx = safetensors_open(path);
  if (!ctx) return false;
  int iv = safetensors_find(ctx, "vertices");
  int ifc = safetensors_find(ctx, "faces");
  if (iv < 0 || ifc < 0) {
    fprintf(stderr, "mesh_io: missing 'vertices' or 'faces' in %s\n", path);
    safetensors_close(ctx);
    return false;
  }
  if (std::strcmp(safetensors_dtype(ctx, iv), "F32") != 0) {
    fprintf(stderr, "mesh_io: 'vertices' must be F32\n");
    safetensors_close(ctx);
    return false;
  }
  if (std::strcmp(safetensors_dtype(ctx, ifc), "I32") != 0) {
    fprintf(stderr, "mesh_io: 'faces' must be I32\n");
    safetensors_close(ctx);
    return false;
  }
  const uint64_t* vs = safetensors_shape(ctx, iv);
  const uint64_t* fs = safetensors_shape(ctx, ifc);
  if (safetensors_ndims(ctx, iv) != 2 || vs[1] != 3) {
    fprintf(stderr, "mesh_io: 'vertices' shape must be [V,3]\n");
    safetensors_close(ctx);
    return false;
  }
  if (safetensors_ndims(ctx, ifc) != 2 || fs[1] != 3) {
    fprintf(stderr, "mesh_io: 'faces' shape must be [F,3]\n");
    safetensors_close(ctx);
    return false;
  }
  uint32_t nv = (uint32_t)vs[0];
  uint32_t nf = (uint32_t)fs[0];
  m.set((const float*)safetensors_data(ctx, iv), nv,
        (const int32_t*)safetensors_data(ctx, ifc), nf);
  safetensors_close(ctx);
  return true;
}

struct MeshWriteOpts {
  const float*   normals = nullptr;  // [V*3]
  const float*   uvs     = nullptr;  // [V*2]
  const int32_t* vmap    = nullptr;  // [V]
};

inline bool saveMesh(const char* path, const Mesh& m, const MeshWriteOpts& opts = {}) {
  stw_writer* w = stw_create();
  if (!w) return false;
  uint32_t nv = m.numV();
  uint32_t nf = m.numF();
  uint64_t vshape[2] = { (uint64_t)nv, 3 };
  uint64_t fshape[2] = { (uint64_t)nf, 3 };
  uint64_t uvshape[2] = { (uint64_t)nv, 2 };
  uint64_t vmshape[1] = { (uint64_t)nv };
  int rc = 0;
  rc |= stw_add(w, "vertices", "F32", vshape, 2, m.v.data(), m.v.size() * sizeof(float));
  rc |= stw_add(w, "faces",    "I32", fshape, 2, m.f.data(), m.f.size() * sizeof(int32_t));
  if (opts.normals)
    rc |= stw_add(w, "vertex_normals", "F32", vshape, 2,
                  opts.normals, (size_t)nv * 3 * sizeof(float));
  if (opts.uvs)
    rc |= stw_add(w, "uvs", "F32", uvshape, 2,
                  opts.uvs, (size_t)nv * 2 * sizeof(float));
  if (opts.vmap)
    rc |= stw_add(w, "vmap", "I32", vmshape, 1,
                  opts.vmap, (size_t)nv * sizeof(int32_t));
  if (rc == 0) rc = stw_save(w, path);
  stw_destroy(w);
  return rc == 0;
}

}  // namespace trellis2
