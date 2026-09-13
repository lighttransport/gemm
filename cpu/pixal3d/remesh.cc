/* CPU port of CuMesh's narrow-band UDF/simple dual contouring algorithm.
 * Uses the shared native closest-point BVH. See ref/pixal3d sources manifest. */
#include "mesh.hh"
#include <unordered_map>

namespace px {
using Coord = std::array<int, 3>;
static uint64_t pack(const Coord &p) {
    return uint64_t(p[0]) | (uint64_t(p[1]) << 21) | (uint64_t(p[2]) << 42);
}
static const int offsets[3][4][3] = {{{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}},
                                     {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}},
                                     {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}}};
Mesh remesh(const Mesh &, const trellis2::ClosestPointBVH &bvh, int resolution) {
    require(resolution >= 32 && (resolution & (resolution - 1)) == 0,
            "Remesh resolution must be power of two >=32");
    // Pixal3D's explicit AABB is [-.5,.5]^3, not the mesh's tight bounds.
    float scale = float(resolution + 3) / resolution, eps = scale / resolution;
    std::vector<Coord> coords;
    coords.reserve(32768);
    for (int x = 0; x < 32; ++x)
        for (int y = 0; y < 32; ++y)
            for (int z = 0; z < 32; ++z)
                coords.push_back({x, y, z});
    for (int level = 32;; level *= 2) {
        std::vector<uint8_t> active(coords.size());
#pragma omp parallel for schedule(dynamic, 256)
        for (size_t i = 0; i < coords.size(); ++i) {
            lightrt::Vec3 p((float(coords[i][0]) + .5f) / level * scale - .5f * scale,
                            (float(coords[i][1]) + .5f) / level * scale - .5f * scale,
                            (float(coords[i][2]) + .5f) / level * scale - .5f * scale);
            active[i] = std::fabs(bvh.query(p).distance - eps) < .87f * scale / level;
        }
        std::vector<Coord> next;
        for (size_t i = 0; i < coords.size(); ++i)
            if (active[i]) {
                if (level == resolution)
                    next.push_back(coords[i]);
                else
                    for (int s = 0; s < 8; ++s)
                        next.push_back({2 * coords[i][0] + (s >> 2), 2 * coords[i][1] + ((s >> 1) & 1),
                                        2 * coords[i][2] + (s & 1)});
            }
        coords.swap(next);
        std::fprintf(stderr, "Pixal3D remesh grid %d: %zu active candidates\n", level, coords.size());
        require(!coords.empty(), "Remesh found no surface");
        if (level == resolution)
            break;
    }
    std::unordered_map<uint64_t, int> corners;
    corners.reserve(coords.size() * 4);
    std::vector<Coord> unique;
    for (const auto &p : coords)
        for (int s = 0; s < 8; ++s) {
            Coord q = {p[0] + (s >> 2), p[1] + ((s >> 1) & 1), p[2] + (s & 1)};
            if (corners.emplace(pack(q), int(unique.size())).second)
                unique.push_back(q);
        }
    Vec udf(unique.size());
#pragma omp parallel for schedule(dynamic, 256)
    for (size_t i = 0; i < unique.size(); ++i) {
        const auto &c = unique[i];
        udf[i] = bvh.query(lightrt::Vec3((float(c[0]) / resolution - .5f) * scale,
                                         (float(c[1]) / resolution - .5f) * scale,
                                         (float(c[2]) / resolution - .5f) * scale))
                     .distance -
                 eps;
    }
    Vec dual(coords.size() * 3);
    std::vector<int> intersected(coords.size() * 3);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < coords.size(); ++i) {
        const Coord &p = coords[i];
        float sum[3] = {};
        int count = 0;
        for (int axis = 0; axis < 3; ++axis)
            for (int u = 0; u < 2; ++u)
                for (int v = 0; v < 2; ++v) {
                    Coord a = p;
                    int orth = 0;
                    for (int d = 0; d < 3; ++d)
                        if (d != axis)
                            a[d] += orth++ == 0 ? u : v;
                    Coord b = a;
                    ++b[axis];
                    float f0 = udf[corners.at(pack(a))], f1 = udf[corners.at(pack(b))];
                    if ((f0 < 0) != (f1 < 0)) {
                        float t = -f0 / (f1 - f0);
                        for (int d = 0; d < 3; ++d)
                            sum[d] += a[d] + (d == axis ? t : 0);
                        ++count;
                        if (u == 1 && v == 1)
                            intersected[i * 3 + axis] = f0 < 0 ? 1 : -1;
                    }
                }
        for (int d = 0; d < 3; ++d)
            dual[i * 3 + d] = ((count ? sum[d] / count : p[d] + .5f) / resolution - .5f) * scale;
    }
    std::unordered_map<uint64_t, int> voxels;
    voxels.reserve(coords.size() * 2);
    for (size_t i = 0; i < coords.size(); ++i)
        voxels.emplace(pack(coords[i]), int(i));
    Mesh result;
    result.v = std::move(dual);
    for (size_t i = 0; i < coords.size(); ++i)
        for (int axis = 0; axis < 3; ++axis)
            if (intersected[i * 3 + axis]) {
                int q[4];
                bool valid = true;
                for (int j = 0; j < 4; ++j) {
                    Coord p = coords[i];
                    for (int d = 0; d < 3; ++d)
                        p[d] += offsets[axis][j][d];
                    auto it = voxels.find(pack(p));
                    if (it == voxels.end()) {
                        valid = false;
                        break;
                    }
                    q[j] = it->second;
                }
                if (!valid)
                    continue;
                const int split1p[6] = {0, 2, 1, 0, 3, 2}, split1n[6] = {0, 1, 2, 0, 2, 3};
                const int split2p[6] = {0, 3, 1, 3, 2, 1}, split2n[6] = {0, 1, 3, 3, 1, 2};
                const int *s1 = intersected[i * 3 + axis] == 1 ? split1p : split1n,
                          *s2 = intersected[i * 3 + axis] == 1 ? split2p : split2n;
                // Keep upstream's exact alignment expression, including indices 1,2,3.
                auto alignment = [&](const int *s) {
                    float a[3], b[3], c[3], d[3], n0[3], n1[3];
                    const float *p0 = &result.v[q[s[0]] * 3], *p1 = &result.v[q[s[1]] * 3],
                                *p2 = &result.v[q[s[2]] * 3], *p3 = &result.v[q[s[3]] * 3];
                    trellis2::v3sub(p1, p0, a);
                    trellis2::v3sub(p2, p0, b);
                    trellis2::v3sub(p2, p1, c);
                    trellis2::v3sub(p3, p1, d);
                    trellis2::v3cross(a, b, n0);
                    trellis2::v3cross(c, d, n1);
                    return std::fabs(trellis2::v3dot(n0, n1));
                };
                const int *split = alignment(s1) > alignment(s2) ? s1 : s2;
                for (int j = 0; j < 6; ++j)
                    result.f.push_back(q[split[j]]);
            }
    std::vector<int> map(result.numV(), -1);
    for (int i : result.f)
        map[i] = 0;
    Vec vertices;
    for (size_t i = 0; i < map.size(); ++i)
        if (map[i] == 0) {
            map[i] = int(vertices.size() / 3);
            vertices.insert(vertices.end(), result.v.begin() + i * 3, result.v.begin() + i * 3 + 3);
        }
    for (int &i : result.f)
        i = map[i];
    result.v.swap(vertices);
    require(result.numF() > 0, "Remesh produced no faces");
    return result;
}
} // namespace px
