// uvUnwrap implementation: thin wrapper around vendored xatlas.
//
// All other ops are header-only in mesh_ops.h. This .cc exists so we don't
// pull <xatlas.h>/<xatlas.cc> into every translation unit.

#include "mesh_ops.h"
#include "xatlas.h"

#include <cstring>

namespace trellis2 {

bool uvUnwrap(const Mesh& in,
              std::vector<float>& out_v,
              std::vector<int32_t>& out_f,
              std::vector<float>& out_uv,
              std::vector<int32_t>& out_vmap) {
  out_v.clear();
  out_f.clear();
  out_uv.clear();
  out_vmap.clear();
  if (in.numV() == 0 || in.numF() == 0) return false;

  xatlas::Atlas* atlas = xatlas::Create();
  if (!atlas) return false;

  xatlas::MeshDecl decl;
  decl.vertexCount = in.numV();
  decl.vertexPositionData = in.v.data();
  decl.vertexPositionStride = 3u * sizeof(float);
  decl.indexCount = in.numF() * 3u;
  decl.indexData = in.f.data();
  decl.indexFormat = xatlas::IndexFormat::UInt32;  // int32 same layout

  xatlas::AddMeshError err = xatlas::AddMesh(atlas, decl, 1);
  if (err != xatlas::AddMeshError::Success) {
    xatlas::Destroy(atlas);
    return false;
  }
  xatlas::AddMeshJoin(atlas);

  xatlas::ChartOptions chart_opts;
  xatlas::PackOptions pack_opts;
  pack_opts.padding = 2;
  pack_opts.blockAlign = true;

  xatlas::Generate(atlas, chart_opts, pack_opts);
  if (atlas->meshCount < 1 || atlas->width == 0 || atlas->height == 0) {
    xatlas::Destroy(atlas);
    return false;
  }
  const xatlas::Mesh& m = atlas->meshes[0];
  if (m.vertexCount == 0 || m.indexCount == 0) {
    xatlas::Destroy(atlas);
    return false;
  }

  const float inv_w = 1.0f / (float)atlas->width;
  const float inv_h = 1.0f / (float)atlas->height;
  out_v.resize(3u * m.vertexCount);
  out_uv.resize(2u * m.vertexCount);
  out_vmap.resize(m.vertexCount);
  for (uint32_t i = 0; i < m.vertexCount; ++i) {
    const xatlas::Vertex& v = m.vertexArray[i];
    if (v.xref >= in.numV()) {
      xatlas::Destroy(atlas);
      out_v.clear(); out_f.clear(); out_uv.clear(); out_vmap.clear();
      return false;
    }
    out_vmap[i] = (int32_t)v.xref;
    out_v[3 * i + 0] = in.v[3 * v.xref + 0];
    out_v[3 * i + 1] = in.v[3 * v.xref + 1];
    out_v[3 * i + 2] = in.v[3 * v.xref + 2];
    out_uv[2 * i + 0] = v.uv[0] * inv_w;
    out_uv[2 * i + 1] = v.uv[1] * inv_h;
  }
  out_f.resize(m.indexCount);
  for (uint32_t i = 0; i < m.indexCount; ++i) {
    out_f[i] = (int32_t)m.indexArray[i];
  }

  xatlas::Destroy(atlas);
  return true;
}

}  // namespace trellis2
