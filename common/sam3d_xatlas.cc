#include "sam3d_xatlas.h"
#include "xatlas.h"

#include <cstdlib>
#include <cstring>

int sam3d_xatlas_generate(const float *positions, int n_positions,
                          const int *triangles, int n_triangles,
                          int atlas_resolution,
                          sam3d_xatlas_mesh *out)
{
    if (!positions || !triangles || !out || n_positions <= 0 ||
        n_triangles <= 0) {
        return -1;
    }
    std::memset(out, 0, sizeof(*out));
    for (int i = 0; i < n_triangles * 3; i++) {
        if (triangles[i] < 0 || triangles[i] >= n_positions) return -1;
    }

    xatlas::Atlas *atlas = xatlas::Create();
    if (!atlas) return -1;

    xatlas::MeshDecl decl;
    decl.vertexCount = (uint32_t)n_positions;
    decl.vertexPositionData = positions;
    decl.vertexPositionStride = 3u * sizeof(float);
    decl.indexCount = (uint32_t)n_triangles * 3u;
    decl.indexData = triangles;
    decl.indexFormat = xatlas::IndexFormat::UInt32;

    xatlas::AddMeshError err = xatlas::AddMesh(atlas, decl, 1);
    if (err != xatlas::AddMeshError::Success) {
        xatlas::Destroy(atlas);
        return -1;
    }
    xatlas::AddMeshJoin(atlas);

    xatlas::ChartOptions chart_opts;
    xatlas::PackOptions pack_opts;
    pack_opts.padding = 2;
    pack_opts.blockAlign = true;
    pack_opts.bruteForce = false;
    if (atlas_resolution > 0) pack_opts.resolution = (uint32_t)atlas_resolution;

    xatlas::Generate(atlas, chart_opts, pack_opts);
    if (atlas->meshCount < 1 || atlas->width == 0 || atlas->height == 0 ||
        atlas->meshes[0].vertexCount == 0 || atlas->meshes[0].indexCount == 0) {
        xatlas::Destroy(atlas);
        return -1;
    }

    const xatlas::Mesh &mesh = atlas->meshes[0];
    sam3d_xatlas_mesh tmp;
    std::memset(&tmp, 0, sizeof(tmp));
    tmp.n_vertices = (int)mesh.vertexCount;
    tmp.n_indices = (int)mesh.indexCount;
    tmp.atlas_width = (int)atlas->width;
    tmp.atlas_height = (int)atlas->height;
    tmp.chart_count = (int)atlas->chartCount;
    tmp.positions = (float *)std::malloc((size_t)tmp.n_vertices * 3 * sizeof(float));
    tmp.uvs = (float *)std::malloc((size_t)tmp.n_vertices * 2 * sizeof(float));
    tmp.xrefs = (uint32_t *)std::malloc((size_t)tmp.n_vertices * sizeof(uint32_t));
    tmp.indices = (uint32_t *)std::malloc((size_t)tmp.n_indices * sizeof(uint32_t));
    if (!tmp.positions || !tmp.uvs || !tmp.xrefs || !tmp.indices) {
        sam3d_xatlas_free(&tmp);
        xatlas::Destroy(atlas);
        return -1;
    }

    const float inv_w = atlas->width ? 1.0f / (float)atlas->width : 1.0f;
    const float inv_h = atlas->height ? 1.0f / (float)atlas->height : 1.0f;
    for (int i = 0; i < tmp.n_vertices; i++) {
        const xatlas::Vertex &v = mesh.vertexArray[i];
        if (v.xref >= (uint32_t)n_positions) {
            sam3d_xatlas_free(&tmp);
            xatlas::Destroy(atlas);
            return -1;
        }
        tmp.xrefs[i] = v.xref;
        tmp.positions[i * 3 + 0] = positions[(size_t)v.xref * 3 + 0];
        tmp.positions[i * 3 + 1] = positions[(size_t)v.xref * 3 + 1];
        tmp.positions[i * 3 + 2] = positions[(size_t)v.xref * 3 + 2];
        tmp.uvs[i * 2 + 0] = v.uv[0] * inv_w;
        tmp.uvs[i * 2 + 1] = v.uv[1] * inv_h;
    }
    std::memcpy(tmp.indices, mesh.indexArray,
                (size_t)tmp.n_indices * sizeof(uint32_t));

    *out = tmp;
    xatlas::Destroy(atlas);
    return 0;
}

void sam3d_xatlas_free(sam3d_xatlas_mesh *m)
{
    if (!m) return;
    std::free(m->positions);
    std::free(m->uvs);
    std::free(m->xrefs);
    std::free(m->indices);
    std::memset(m, 0, sizeof(*m));
}
