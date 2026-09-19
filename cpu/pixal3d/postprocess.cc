#include "../../common/safetensors_writer.h"
#include "../../common/trellis2_fdg_mesh.h"
#include "mesh.hh"
#include <filesystem>
#include <opencv2/photo.hpp>
#include <parallel/algorithm>
#include <unordered_map>
#include <unordered_set>

void px_opencv_inpaint(cv::InputArray, cv::InputArray, cv::OutputArray, double, int);

namespace px {
static uint64_t packed(int x, int y, int z) {
    return uint64_t(x) | (uint64_t(y) << 21) | (uint64_t(z) << 42);
}
void fill_holes(Mesh &m) {
    std::vector<uint64_t> edges;
    edges.reserve(m.f.size());
    for (size_t f = 0; f < m.f.size(); f += 3)
        for (int j = 0; j < 3; ++j) {
            int a = m.f[f + j], b = m.f[f + (j + 1) % 3];
            if (a > b)
                std::swap(a, b);
            edges.push_back((uint64_t(a) << 32) | uint32_t(b));
        }
    __gnu_parallel::sort(edges.begin(), edges.end());
    std::vector<std::vector<int>> adjacency(m.numV());
    for (size_t i = 0; i < edges.size();) {
        size_t j = i + 1;
        while (j < edges.size() && edges[j] == edges[i])
            ++j;
        if (j == i + 1) {
            int a = int(edges[i] >> 32), b = uint32_t(edges[i]);
            adjacency[a].push_back(b);
            adjacency[b].push_back(a);
        }
        i = j;
    }
    edges.clear();
    edges.shrink_to_fit();
    std::unordered_set<int> visited;
    for (size_t start_index = 0; start_index < adjacency.size(); ++start_index) {
        int start = int(start_index);
        if (visited.count(start) || adjacency[start].size() != 2)
            continue;
        std::vector<int> loop;
        int last = -1, current = start;
        bool valid = true;
        do {
            if (visited.count(current) || adjacency[current].size() != 2) {
                valid = false;
                break;
            }
            visited.insert(current);
            loop.push_back(current);
            const auto &next = adjacency[current];
            int id = next[0] == last ? next[1] : next[0];
            last = current;
            current = id;
        } while (current != start);
        if (!valid || loop.size() < 3)
            continue;
        float perimeter = 0;
        float center[3] = {};
        for (size_t i = 0; i < loop.size(); ++i) {
            float length = 0;
            for (int c = 0; c < 3; ++c) {
                float d = m.v[3 * loop[i] + c] - m.v[3 * loop[(i + 1) % loop.size()] + c];
                length += d * d;
                center[c] += m.v[3 * loop[i] + c];
            }
            perimeter += std::sqrt(length);
        }
        if (perimeter >= .03f)
            continue;
        int id = int(m.numV());
        for (int c = 0; c < 3; ++c)
            m.v.push_back(center[c] / loop.size());
        for (size_t i = 0; i < loop.size(); ++i) {
            int a = loop[i], b = loop[(i + 1) % loop.size()];
            if (a > b)
                std::swap(a, b);
            m.f.insert(m.f.end(), {a, b, id});
        }
    }
}
void inpaint(std::vector<uint8_t> &image, int channels, const std::vector<uint8_t> &mask, int size,
             int radius) {
    cv::Mat input(size, size, CV_MAKETYPE(CV_8U, channels), image.data()),
        missing(size, size, CV_8UC1, const_cast<uint8_t *>(mask.data())), output;
    std::fprintf(stderr, "Pixal3D inpaint: %d channels, radius %d\n", channels, radius);
    px_opencv_inpaint(input, missing, output, radius, cv::INPAINT_TELEA);
    std::memcpy(image.data(), output.data, image.size());
}
void vertex_normals(const Mesh &mesh, Vec &normals) {
    trellis2::computeVertexNormals(mesh, normals);
    // Opposing incident faces can cancel exactly. CuMesh falls back to its first
    // incident face; normalize that direction too, as required by glTF NORMAL.
    for (size_t f = 0; f < mesh.f.size(); f += 3) {
        const float *a = mesh.v.data() + 3 * mesh.f[f], *b = mesh.v.data() + 3 * mesh.f[f + 1],
                    *c = mesh.v.data() + 3 * mesh.f[f + 2];
        float e0[3], e1[3], face[3];
        trellis2::v3sub(b, a, e0);
        trellis2::v3sub(c, a, e1);
        trellis2::v3cross(e0, e1, face);
        double length =
            std::sqrt(double(face[0]) * face[0] + double(face[1]) * face[1] + double(face[2]) * face[2]);
        if (!(length > 0))
            continue;
        for (int j = 0; j < 3; ++j) {
            float *normal = normals.data() + 3 * mesh.f[f + j];
            if (normal[0] == 0 && normal[1] == 0 && normal[2] == 0)
                for (int k = 0; k < 3; ++k)
                    normal[k] = float(face[k] / length);
        }
    }
    for (size_t i = 0; i < normals.size(); i += 3) {
        float length = trellis2::v3dot(normals.data() + i, normals.data() + i);
        require(std::isfinite(length) && length > 0, "Cannot produce a finite surface normal");
    }
}
template <typename T> static T *copy_result(const std::vector<T> &x) {
    T *out = static_cast<T *>(std::malloc(x.size() * sizeof(T)));
    require(out || x.empty(), "Cannot allocate mesh result");
    std::copy(x.begin(), x.end(), out);
    return out;
}
static void dump_mesh(const pixal3d_options &options, const char *name, const Mesh &mesh) {
    if (!options.dump_dir || !*options.dump_dir)
        return;
    std::filesystem::create_directories(options.dump_dir);
    std::unique_ptr<stw_writer, decltype(&stw_destroy)> writer(stw_create(), stw_destroy);
    require(bool(writer), "Cannot create mesh dump writer");
    uint64_t vertices[2] = {mesh.numV(), 3}, faces[2] = {mesh.numF(), 3};
    require(stw_add(writer.get(), "vertices", "F32", vertices, 2, mesh.v.data(), mesh.v.size() * 4) == 0 &&
                stw_add(writer.get(), "faces", "I32", faces, 2, mesh.f.data(), mesh.f.size() * 4) == 0,
            "Cannot add mesh dump tensors");
    auto path = std::string(options.dump_dir) + "/" + name + ".safetensors";
    require(stw_save(writer.get(), path.c_str()) == 0, "Cannot save mesh dump: " + path);
}
void postprocess(const Sparse &shape, const Sparse &texture, const pixal3d_options &options,
                 pixal3d_result &result, Engine *profile) {
    auto phase = std::chrono::steady_clock::now();
    auto mark = [&](const char *name) {
        auto now = std::chrono::steady_clock::now();
        if (profile)
            profile->record(std::string("postprocess.") + name,
                            std::chrono::duration<double>(now - phase).count());
        phase = now;
    };
    require(shape.channels == 7 && texture.channels == 6 && shape.coords == texture.coords,
            "Invalid shape/texture decoder outputs");
    Vec decoded = shape.feats;
    for (int i = 0; i < shape.rows(); ++i) {
        for (int c = 0; c < 3; ++c)
            decoded[size_t(i) * 7 + c] = 2 / (1 + std::exp(-decoded[size_t(i) * 7 + c])) - .5f;
        float v = decoded[size_t(i) * 7 + 6];
        decoded[size_t(i) * 7 + 6] = v > 20 ? v : std::log1p(std::exp(v));
    }
    const float aabb[6] = {-.5f, -.5f, -.5f, .5f, .5f, .5f};
    auto extracted = t2_fdg_to_mesh_bxyz(shape.coords.data(), decoded.data(), shape.rows(), 1.f / 1024, aabb);
    Mesh original;
    original.set(extracted.vertices, extracted.n_verts, extracted.triangles, extracted.n_tris);
    t2_fdg_mesh_free(&extracted);
    require(original.numF() > 0, "FDG extraction produced no faces");
    mark("fdg");
    fill_holes(original);
    dump_mesh(options, "mesh_fdg", original);
    std::fprintf(stderr, "Pixal3D FDG: %u vertices, %u faces\n", original.numV(), original.numF());
    trellis2::ClosestPointBVH bvh;
    require(bvh.build(original.v.data(), original.numV(), original.f.data(), original.numF()),
            "Cannot build original mesh BVH");
    mark("holes_bvh");
    Mesh mesh = remesh(original, bvh);
    mark("remesh");
    dump_mesh(options, "mesh_remeshed", mesh);
    simplify(mesh, options.decimation_target);
    clean_for_uv(mesh);
    mark("simplify");
    dump_mesh(options, "mesh_simplified", mesh);
    Vec vertices, uv, normals;
    std::vector<int32_t> faces, vmap;
    unwrap(mesh, vertices, faces, uv, vmap);
    Vec original_normals;
    vertex_normals(mesh, original_normals);
    normals.resize(vertices.size());
    for (size_t i = 0; i < vmap.size(); ++i)
        std::copy_n(original_normals.data() + size_t(vmap[i]) * 3, 3, normals.data() + i * 3);
    mark("unwrap_normals");
    int size = options.texture_size;
    size_t pixels = size_t(size) * size;
    std::vector<int> raster(pixels, -1);
    Vec bary(pixels * 2);
    for (size_t f = 0; f < faces.size() / 3; ++f) {
        int ids[3] = {faces[3 * f], faces[3 * f + 1], faces[3 * f + 2]};
        float p[3][2];
        for (int i = 0; i < 3; ++i)
            for (int c = 0; c < 2; ++c)
                p[i][c] = uv[size_t(ids[i]) * 2 + c] * size;
        float det = (p[1][1] - p[2][1]) * (p[0][0] - p[2][0]) + (p[2][0] - p[1][0]) * (p[0][1] - p[2][1]);
        if (std::fabs(det) < 1e-12f)
            continue;
        int x0 = std::max(0, int(std::ceil(std::min({p[0][0], p[1][0], p[2][0]}) - .5f))),
            x1 = std::min(size - 1, int(std::floor(std::max({p[0][0], p[1][0], p[2][0]}) - .5f)));
        int y0 = std::max(0, int(std::ceil(std::min({p[0][1], p[1][1], p[2][1]}) - .5f))),
            y1 = std::min(size - 1, int(std::floor(std::max({p[0][1], p[1][1], p[2][1]}) - .5f)));
        for (int y = y0; y <= y1; ++y)
            for (int x = x0; x <= x1; ++x) {
                float u =
                    ((p[1][1] - p[2][1]) * (x + .5f - p[2][0]) + (p[2][0] - p[1][0]) * (y + .5f - p[2][1])) /
                    det;
                float v =
                    ((p[2][1] - p[0][1]) * (x + .5f - p[2][0]) + (p[0][0] - p[2][0]) * (y + .5f - p[2][1])) /
                    det;
                if (u >= 0 && v >= 0 && u + v <= 1) {
                    size_t id = size_t(y) * size + x;
                    raster[id] = int(f);
                    bary[2 * id] = u;
                    bary[2 * id + 1] = v;
                }
            }
    }
    mark("raster");
    std::unordered_map<uint64_t, int> index;
    index.reserve(texture.rows() * 2);
    for (int i = 0; i < texture.rows(); ++i)
        index.emplace(packed(texture.coords[4 * i + 1], texture.coords[4 * i + 2], texture.coords[4 * i + 3]),
                      i);
    std::vector<uint8_t> base(pixels * 3), metal(pixels), rough(pixels), alpha(pixels), missing(pixels, 1);
    std::fprintf(stderr, "Pixal3D bake: %d x %d PBR textures\n", size, size);
#pragma omp parallel for schedule(dynamic, 256)
    for (size_t id = 0; id < pixels; ++id) {
        int f = raster[id];
        if (f < 0)
            continue;
        missing[id] = 0;
        float pos[3] = {};
        float weights[3] = {bary[2 * id], bary[2 * id + 1], 1 - bary[2 * id] - bary[2 * id + 1]};
        for (int j = 0; j < 3; ++j)
            for (int c = 0; c < 3; ++c)
                pos[c] += weights[j] * vertices[size_t(faces[3 * f + j]) * 3 + c];
        auto hit = bvh.query(lightrt::Vec3(pos[0], pos[1], pos[2]));
        float q[3];
        int low[3];
        for (int c = 0; c < 3; ++c) {
            q[c] = (hit.closest[c] + .5f) * 1024;
            low[c] = int(std::floor(q[c] - .5f));
        }
        float attr[6] = {}, sum = 0;
        for (int s = 0; s < 8; ++s) {
            int x = low[0] + (s & 1), y = low[1] + ((s >> 1) & 1), z = low[2] + ((s >> 2) & 1);
            if (x < 0 || x >= 1024 || y < 0 || y >= 1024 || z < 0 || z >= 1024)
                continue;
            auto it = index.find(packed(x, y, z));
            if (it == index.end())
                continue;
            float w = (1 - std::fabs(q[0] - x - .5f)) * (1 - std::fabs(q[1] - y - .5f)) *
                      (1 - std::fabs(q[2] - z - .5f));
            sum += w;
            for (int c = 0; c < 6; ++c)
                attr[c] += w * texture.feats[size_t(it->second) * 6 + c];
        }
        uint8_t color[6];
        for (int c = 0; c < 6; ++c)
            color[c] = uint8_t(std::clamp(attr[c] / std::max(sum, 1e-12f) * 255, 0.f, 255.f));
        for (int c = 0; c < 3; ++c)
            base[id * 3 + c] = color[c];
        metal[id] = color[3];
        rough[id] = color[4];
        alpha[id] = color[5];
    }
    mark("bake");
    std::vector<uint8_t> material(pixels * 3);
    for (size_t i = 0; i < pixels; ++i) {
        material[3 * i] = metal[i];
        material[3 * i + 1] = rough[i];
        material[3 * i + 2] = alpha[i];
    }
    // The two Telea solves share only the immutable missing-pixel mask.
    // Execute them concurrently without changing either numerical path.
#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
        inpaint(base, 3, missing, size, 3);
#pragma omp section
        inpaint(material, 3, missing, size, 1);
    }
    for (size_t i = 0; i < pixels; ++i) {
        metal[i] = material[3 * i];
        rough[i] = material[3 * i + 1];
        alpha[i] = material[3 * i + 2];
    }
    mark("inpaint");
    std::vector<uint8_t> rgba(pixels * 4), mr(pixels * 3);
    for (size_t i = 0; i < pixels; ++i) {
        std::copy_n(base.data() + i * 3, 3, rgba.data() + i * 4);
        rgba[i * 4 + 3] = alpha[i];
        mr[i * 3 + 1] = rough[i];
        mr[i * 3 + 2] = metal[i];
    }
    // Combined o_voxel GLB and Pixal3D rotations: (x,y,z) -> (-x,y,-z).
    // Trimesh flips V on export, cancelling o_voxel's preceding V flip.
    for (size_t i = 0; i < vertices.size(); i += 3) {
        vertices[i] = -vertices[i];
        vertices[i + 2] = -vertices[i + 2];
        normals[i] = -normals[i];
        normals[i + 2] = -normals[i + 2];
    }
    result.vertices = copy_result(vertices);
    result.normals = copy_result(normals);
    result.uvs = copy_result(uv);
    result.triangles = reinterpret_cast<uint32_t *>(copy_result(faces));
    result.base_color_rgba = copy_result(rgba);
    result.metallic_roughness_rgb = copy_result(mr);
    result.vertex_count = int(vertices.size() / 3);
    result.triangle_count = int(faces.size() / 3);
    result.texture_size = size;
}
} // namespace px
