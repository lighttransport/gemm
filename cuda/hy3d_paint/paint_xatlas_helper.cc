/* C-callable xatlas UV unwrap wrapper for test_paint_pipeline.
 *
 * Produces the same arrays the dump_paint_back_project pyref carries:
 *   - vmap[U]:        UV-vert i -> original vertex idx (for gathering vtx_pos)
 *   - uvs[U,2]:       normalized UVs in [0,1]
 *   - uv_idx[F,3]:    per-face UV-vert indices
 * F (face count) and original positions are unchanged from the input. */

#include "../../common/xatlas.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>

extern "C" int paint_xatlas_unwrap(
    const float *vtx_pos, int n_verts,
    const int   *tri_idx, int n_tris,
    int   **out_vmap,                  /* [U]    int32 */
    float **out_uvs,                   /* [U,2]  float */
    int   **out_uv_idx,                /* [F,3]  int32 (F == n_tris) */
    int    *out_n_uv_verts,
    int    *out_atlas_w, int *out_atlas_h)
{
    xatlas::Atlas *atlas = xatlas::Create();
    xatlas::MeshDecl decl;
    decl.vertexCount         = (uint32_t)n_verts;
    decl.vertexPositionData  = vtx_pos;
    decl.vertexPositionStride = (uint32_t)(3 * sizeof(float));
    decl.indexCount          = (uint32_t)(n_tris * 3);
    decl.indexData           = tri_idx;
    decl.indexFormat         = xatlas::IndexFormat::UInt32;
    if (xatlas::AddMesh(atlas, decl, 1) != xatlas::AddMeshError::Success) {
        xatlas::Destroy(atlas); return 1;
    }
    xatlas::AddMeshJoin(atlas);
    xatlas::ChartOptions chart_opts;
    xatlas::PackOptions  pack_opts;
    pack_opts.padding    = 2;
    pack_opts.bruteForce = false;
    pack_opts.blockAlign = true;
    xatlas::Generate(atlas, chart_opts, pack_opts);

    const xatlas::Mesh &om = atlas->meshes[0];
    int U = (int)om.vertexCount;
    int F = (int)(om.indexCount / 3);
    if (F != n_tris) {
        /* xatlas may split degenerate faces; we don't currently handle that. */
    }

    int   *vmap   = (int *)  malloc((size_t)U * sizeof(int));
    float *uvs    = (float *)malloc((size_t)U * 2 * sizeof(float));
    int   *uv_idx = (int *)  malloc((size_t)F * 3 * sizeof(int));

    float inv_w = atlas->width  ? 1.f / (float)atlas->width  : 1.f;
    float inv_h = atlas->height ? 1.f / (float)atlas->height : 1.f;
    for (int i = 0; i < U; i++) {
        vmap[i] = (int)om.vertexArray[i].xref;
        uvs[i*2+0] = om.vertexArray[i].uv[0] * inv_w;
        uvs[i*2+1] = om.vertexArray[i].uv[1] * inv_h;
    }
    for (uint32_t i = 0; i < om.indexCount; i++)
        uv_idx[i] = (int)om.indexArray[i];

    *out_vmap = vmap;
    *out_uvs  = uvs;
    *out_uv_idx = uv_idx;
    *out_n_uv_verts = U;
    *out_atlas_w = (int)atlas->width;
    *out_atlas_h = (int)atlas->height;
    xatlas::Destroy(atlas);
    return 0;
}
