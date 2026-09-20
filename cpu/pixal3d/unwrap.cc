/* CuMesh normal-cone chart merging, followed by the shared native xatlas. */
#include "mesh.hh"
#include "../../common/xatlas.h"
#include <numeric>
#include <parallel/algorithm>
namespace px {
using V3 = lightrt::Vec3;
static V3 point(const Mesh &m, int i) { return V3(m.v[3 * i], m.v[3 * i + 1], m.v[3 * i + 2]); }
struct EdgeFace {
    uint64_t edge;
    int face;
};
struct Adjacency {
    int a, b;
    float length;
};
// Match CuMesh.uv_unwrap's cleanup, including its AND threshold rule.
void clean_for_uv(Mesh &mesh) {
    std::vector<int32_t> faces;
    for (size_t i = 0; i < mesh.f.size(); i += 3) {
        int a = mesh.f[i], b = mesh.f[i + 1], c = mesh.f[i + 2];
        if (a == b || b == c || c == a)
            continue;
        V3 e0 = point(mesh, b) - point(mesh, a), e1 = point(mesh, c) - point(mesh, b),
           e2 = point(mesh, a) - point(mesh, c);
        float edge = std::sqrt(std::max({e0.dot(e0), e1.dot(e1), e2.dot(e2)}));
        V3 normal = e0.cross(e1);
        float area = std::sqrt(normal.dot(normal)) * .5f;
        if (area < std::min(1e-12f * edge * edge, 1e-24f))
            continue;
        faces.insert(faces.end(), {a, b, c});
    }
    require(!faces.empty(), "UV cleanup removed all faces");
    std::vector<int> map(mesh.numV(), -1);
    for (int i : faces)
        map[i] = 0;
    Vec vertices;
    for (size_t i = 0; i < map.size(); ++i)
        if (map[i] == 0) {
            map[i] = int(vertices.size() / 3);
            vertices.insert(vertices.end(), mesh.v.begin() + 3 * i, mesh.v.begin() + 3 * i + 3);
        }
    for (int &i : faces)
        i = map[i];
    mesh.v.swap(vertices);
    mesh.f.swap(faces);
}
void unwrap(const Mesh &m, Vec &vertices, std::vector<int32_t> &faces, Vec &uv, std::vector<int32_t> &vmap) {
    int nf = int(m.numF()), nc = nf;
    std::vector<V3> normals(nf);
    Vec areas(nf);
    std::vector<EdgeFace> edges;
    edges.reserve(m.f.size());
    for (int f = 0; f < nf; ++f) {
        V3 normal = (point(m, m.f[3 * f + 1]) - point(m, m.f[3 * f]))
                        .cross(point(m, m.f[3 * f + 2]) - point(m, m.f[3 * f]));
        float length = std::sqrt(normal.dot(normal));
        areas[f] = .5f * length;
        normals[f] = normal * (1 / std::max(length, 1e-20f));
        for (int j = 0; j < 3; ++j) {
            int a = m.f[3 * f + j], b = m.f[3 * f + (j + 1) % 3];
            if (a > b)
                std::swap(a, b);
            edges.push_back({(uint64_t(a) << 32) | uint32_t(b), f});
        }
    }
    __gnu_parallel::sort(edges.begin(), edges.end(),
                         [](const auto &a, const auto &b) { return a.edge < b.edge; });
    std::vector<Adjacency> adjacency;
    adjacency.reserve(edges.size() / 2);
    for (size_t i = 0; i < edges.size();) {
        size_t j = i + 1;
        while (j < edges.size() && edges[j].edge == edges[i].edge)
            ++j;
        if (j - i == 2) {
            V3 d = point(m, int(edges[i].edge >> 32)) - point(m, uint32_t(edges[i].edge));
            adjacency.push_back({edges[i].face, edges[i + 1].face, std::sqrt(d.dot(d))});
        }
        i = j;
    }
    edges.clear();
    edges.shrink_to_fit();
    std::vector<int> charts(nf);
    std::iota(charts.begin(), charts.end(), 0);
    for (;;) {
        std::vector<V3> axes(nc, V3(0, 0, 0));
        Vec area(nc), angle(nc), perimeter(nc);
        for (int f = 0; f < nf; ++f) {
            axes[charts[f]] = axes[charts[f]] + normals[f];
            area[charts[f]] += areas[f];
        }
        for (auto &a : axes)
            a = a * (1 / std::max(std::sqrt(a.dot(a)), 1e-20f));
        for (int f = 0; f < nf; ++f)
            angle[charts[f]] =
                std::max(angle[charts[f]], std::acos(std::clamp(axes[charts[f]].dot(normals[f]), -1.f, 1.f)));
        std::vector<std::pair<uint64_t, float>> merged;
        merged.reserve(adjacency.size());
        for (const auto &e : adjacency) {
            int a = charts[e.a], b = charts[e.b];
            if (a == b)
                continue;
            if (a > b)
                std::swap(a, b);
            merged.emplace_back((uint64_t(a) << 32) | uint32_t(b), e.length);
        }
        // Stable grouping retains each chart pair's original adjacency order,
        // and therefore the exact floating-point accumulation used by std::map.
        std::stable_sort(merged.begin(), merged.end(),
                         [](const auto &a, const auto &b) { return a.first < b.first; });
        std::vector<Adjacency> ce;
        ce.reserve(merged.size());
        for (size_t i = 0; i < merged.size();) {
            size_t j = i + 1;
            float length = merged[i].second;
            while (j < merged.size() && merged[j].first == merged[i].first)
                length += merged[j++].second;
            int a = int(merged[i].first >> 32), b = uint32_t(merged[i].first);
            ce.push_back({a, b, length});
            perimeter[a] += length;
            perimeter[b] += length;
            i = j;
        }
        Vec cost(ce.size());
        std::vector<int> best(nc, -1);
        for (size_t i = 0; i < ce.size(); ++i) {
            auto e = ce[i];
            float axis_angle = std::acos(std::clamp(axes[e.a].dot(axes[e.b]), -1.f, 1.f));
            float low = std::min(-angle[e.a], axis_angle - angle[e.b]),
                  high = std::max(angle[e.a], axis_angle + angle[e.b]);
            float ar = area[e.a] + area[e.b], per = perimeter[e.a] + perimeter[e.b] - 2 * e.length;
            cost[i] = (high - low) * .5f + .1f * ar + .0001f * per * per / std::max(ar, 1e-20f);
            for (int c : {e.a, e.b})
                if (best[c] < 0 || cost[i] < cost[best[c]])
                    best[c] = int(i);
        }
        std::vector<int> map(nc);
        std::iota(map.begin(), map.end(), 0);
        int count = 0;
        for (size_t i = 0; i < ce.size(); ++i)
            if (cost[i] <= 1.57079632679f && best[ce[i].a] == int(i) && best[ce[i].b] == int(i)) {
                map[ce[i].b] = ce[i].a;
                ++count;
            }
        if (!count)
            break;
        std::vector<int> compact(nc, -1);
        int next = 0;
        for (int c = 0; c < nc; ++c)
            if (map[c] == c)
                compact[c] = next++;
        for (int &c : charts)
            c = compact[map[c]];
        nc = next;
    }
    std::fprintf(stderr, "Pixal3D UV: %d normal-cone charts\n", nc);
    std::vector<std::vector<int>> chart_faces(nc);
    std::vector<size_t> chart_sizes(nc);
    for (int f = 0; f < nf; ++f)
        ++chart_sizes[charts[f]];
    for (int c = 0; c < nc; ++c)
        chart_faces[c].reserve(chart_sizes[c]);
    for (int f = 0; f < nf; ++f)
        chart_faces[charts[f]].push_back(f);
    std::unique_ptr<xatlas::Atlas, decltype(&xatlas::Destroy)> atlas(xatlas::Create(), xatlas::Destroy);
    require(bool(atlas), "Cannot create UV atlas");
    std::vector<std::vector<int>> maps(nc);
    for (int c = 0; c < nc; ++c) {
        auto &map = maps[c];
        map.reserve(chart_faces[c].size() * 3);
        for (int f : chart_faces[c])
            for (int j = 0; j < 3; ++j)
                map.push_back(m.f[3 * f + j]);
        std::sort(map.begin(), map.end());
        map.erase(std::unique(map.begin(), map.end()), map.end());
        Vec v;
        std::vector<uint32_t> f;
        v.reserve(map.size() * 3);
        f.reserve(chart_faces[c].size() * 3);
        for (int id : map)
            v.insert(v.end(), m.v.begin() + 3 * id, m.v.begin() + 3 * id + 3);
        for (int id : chart_faces[c])
            for (int j = 0; j < 3; ++j)
                f.push_back(
                    uint32_t(std::lower_bound(map.begin(), map.end(), m.f[3 * id + j]) - map.begin()));
        xatlas::MeshDecl decl;
        decl.vertexCount = uint32_t(map.size());
        decl.vertexPositionData = v.data();
        decl.vertexPositionStride = 12;
        decl.indexCount = uint32_t(f.size());
        decl.indexData = f.data();
        decl.indexFormat = xatlas::IndexFormat::UInt32;
        require(xatlas::AddMesh(atlas.get(), decl, nc) == xatlas::AddMeshError::Success,
                "Cannot add UV chart");
    }
    xatlas::AddMeshJoin(atlas.get());
    xatlas::Generate(atlas.get());
    require(atlas->width && atlas->height, "UV atlas is empty");
    require(atlas->meshCount == uint32_t(nc), "UV chart count changed unexpectedly");
    for (int c = 0; c < nc; ++c) {
        const auto &mesh = atlas->meshes[c];
        int base = int(vertices.size() / 3);
        for (uint32_t i = 0; i < mesh.vertexCount; ++i) {
            const auto &v = mesh.vertexArray[i];
            int id = maps[c].at(v.xref);
            vmap.push_back(id);
            vertices.insert(vertices.end(), m.v.begin() + 3 * id, m.v.begin() + 3 * id + 3);
            uv.push_back(v.uv[0] / atlas->width);
            uv.push_back(v.uv[1] / atlas->height);
        }
        for (uint32_t i = 0; i < mesh.indexCount; ++i)
            faces.push_back(base + mesh.indexArray[i]);
    }
}
} // namespace px
