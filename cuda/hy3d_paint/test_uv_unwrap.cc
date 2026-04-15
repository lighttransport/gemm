/*
 * test_uv_unwrap.cc - Standalone xatlas UV unwrap on an OBJ mesh.
 *
 * Loads an OBJ, runs xatlas, and writes a new OBJ with per-face
 * "f v/t v/t v/t" indices referencing the generated UVs. This is the
 * first native C++ building block of the Hunyuan3D-2.1 texture-gen
 * port -- it replaces the Python xatlas call that hy3dpaint uses inside
 * `utils/uvwrap_utils.mesh_uv_wrap` before running the multiview
 * diffusion model.
 *
 * Usage:
 *   ./test_uv_unwrap <input.obj> [<output.obj>]
 *
 * Build:
 *   g++ -O3 -std=c++17 -o test_uv_unwrap test_uv_unwrap.cc \
 *       ../../common/xatlas.cc -lpthread
 */

#include "../../common/xatlas.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct ObjMesh {
    std::vector<float> positions;  // xyz, flat
    std::vector<int>   indices;    // flat, 3 per face
};

static bool load_obj(const char *path, ObjMesh &m) {
    FILE *f = std::fopen(path, "rb");
    if (!f) {
        std::fprintf(stderr, "ERROR: cannot open %s\n", path);
        return false;
    }
    char line[4096];
    while (std::fgets(line, sizeof(line), f)) {
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            if (std::sscanf(line + 2, "%f %f %f", &x, &y, &z) == 3) {
                m.positions.push_back(x);
                m.positions.push_back(y);
                m.positions.push_back(z);
            }
        } else if (line[0] == 'f' && line[1] == ' ') {
            /* Support "f a b c" and "f a/? b/? c/?" (discard '/...' parts). */
            int i0 = 0, i1 = 0, i2 = 0;
            const char *p = line + 2;
            int idx[3] = {0, 0, 0};
            int k = 0;
            while (*p && k < 3) {
                while (*p == ' ' || *p == '\t') p++;
                if (!*p || *p == '\n') break;
                idx[k++] = std::atoi(p);
                while (*p && *p != ' ' && *p != '\t' && *p != '\n') p++;
            }
            if (k == 3) {
                /* OBJ is 1-indexed. Negative values reference from end. */
                for (int i = 0; i < 3; i++) {
                    if (idx[i] < 0)
                        idx[i] = (int)(m.positions.size() / 3) + idx[i];
                    else
                        idx[i] -= 1;
                }
                m.indices.push_back(idx[0]);
                m.indices.push_back(idx[1]);
                m.indices.push_back(idx[2]);
            }
        }
    }
    std::fclose(f);
    if (m.positions.empty() || m.indices.empty()) {
        std::fprintf(stderr, "ERROR: OBJ empty? verts=%zu tris=%zu\n",
                     m.positions.size() / 3, m.indices.size() / 3);
        return false;
    }
    return true;
}

/* Write the unwrapped mesh in standard OBJ "v/vt" form.
 * xatlas's output vertices are per-chart splits of the original verts,
 * so the vertex count grows. Each output vertex carries an `xref` back
 * to the input vertex index (for the position) and a normalized UV. */
static bool write_obj(const char *path, const ObjMesh &in,
                      const xatlas::Atlas *atlas) {
    FILE *f = std::fopen(path, "wb");
    if (!f) {
        std::fprintf(stderr, "ERROR: cannot open %s for write\n", path);
        return false;
    }
    std::fprintf(f, "# UV unwrapped by xatlas (vendored common/xatlas.{h,cc})\n");
    std::fprintf(f, "# atlas %ux%u  meshes=%u  charts=%u\n",
                 atlas->width, atlas->height, atlas->meshCount, atlas->chartCount);

    const xatlas::Mesh &out_mesh = atlas->meshes[0];

    /* Emit one "v" per atlas output vertex so "v" and "vt" share index. */
    for (uint32_t i = 0; i < out_mesh.vertexCount; i++) {
        uint32_t xref = out_mesh.vertexArray[i].xref;
        float x = in.positions[xref * 3 + 0];
        float y = in.positions[xref * 3 + 1];
        float z = in.positions[xref * 3 + 2];
        std::fprintf(f, "v %.6f %.6f %.6f\n", x, y, z);
    }
    /* UVs normalized to [0, 1] by atlas width/height. */
    float inv_w = atlas->width  ? 1.0f / (float)atlas->width  : 1.0f;
    float inv_h = atlas->height ? 1.0f / (float)atlas->height : 1.0f;
    for (uint32_t i = 0; i < out_mesh.vertexCount; i++) {
        float u = out_mesh.vertexArray[i].uv[0] * inv_w;
        float v = out_mesh.vertexArray[i].uv[1] * inv_h;
        std::fprintf(f, "vt %.6f %.6f\n", u, v);
    }
    for (uint32_t i = 0; i < out_mesh.indexCount; i += 3) {
        uint32_t a = out_mesh.indexArray[i + 0] + 1;
        uint32_t b = out_mesh.indexArray[i + 1] + 1;
        uint32_t c = out_mesh.indexArray[i + 2] + 1;
        std::fprintf(f, "f %u/%u %u/%u %u/%u\n", a, a, b, b, c, c);
    }
    std::fclose(f);
    return true;
}

static bool xatlas_progress(xatlas::ProgressCategory category, int progress,
                            void *user) {
    (void)user;
    if (progress == 0)
        std::fprintf(stderr, "  xatlas: %-16s   0%%",
                     xatlas::StringForEnum(category));
    else if (progress == 100)
        std::fprintf(stderr, "\r  xatlas: %-16s 100%%\n",
                     xatlas::StringForEnum(category));
    else if (progress % 10 == 0)
        std::fprintf(stderr, "\r  xatlas: %-16s %3d%%",
                     xatlas::StringForEnum(category), progress);
    return true;  // false to abort
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "Usage: %s <input.obj> [<output.obj>]\n", argv[0]);
        return 1;
    }
    const char *in_path  = argv[1];
    const char *out_path = argc >= 3 ? argv[2] : "uv_unwrapped.obj";

    ObjMesh m;
    if (!load_obj(in_path, m)) return 1;
    std::fprintf(stderr, "Loaded %s: %zu verts, %zu tris\n",
                 in_path, m.positions.size() / 3, m.indices.size() / 3);

    xatlas::Atlas *atlas = xatlas::Create();
    xatlas::SetProgressCallback(atlas, xatlas_progress, nullptr);

    xatlas::MeshDecl decl;
    decl.vertexCount         = (uint32_t)(m.positions.size() / 3);
    decl.vertexPositionData  = m.positions.data();
    decl.vertexPositionStride = (uint32_t)(3 * sizeof(float));
    decl.indexCount          = (uint32_t)m.indices.size();
    decl.indexData           = m.indices.data();
    decl.indexFormat         = xatlas::IndexFormat::UInt32;

    xatlas::AddMeshError err = xatlas::AddMesh(atlas, decl, 1);
    if (err != xatlas::AddMeshError::Success) {
        std::fprintf(stderr, "xatlas::AddMesh failed: %s\n",
                     xatlas::StringForEnum(err));
        xatlas::Destroy(atlas);
        return 1;
    }
    xatlas::AddMeshJoin(atlas);

    xatlas::ChartOptions chart_opts;
    xatlas::PackOptions pack_opts;
    pack_opts.padding = 2;
    pack_opts.bruteForce = false;
    pack_opts.blockAlign = true;
    std::fprintf(stderr, "Running xatlas Generate()...\n");
    xatlas::Generate(atlas, chart_opts, pack_opts);

    std::fprintf(stderr, "Atlas: %ux%u, %u charts, %u output verts\n",
                 atlas->width, atlas->height, atlas->chartCount,
                 atlas->meshes[0].vertexCount);

    if (!write_obj(out_path, m, atlas)) {
        xatlas::Destroy(atlas);
        return 1;
    }
    std::fprintf(stderr, "Wrote %s\n", out_path);
    xatlas::Destroy(atlas);
    return 0;
}
