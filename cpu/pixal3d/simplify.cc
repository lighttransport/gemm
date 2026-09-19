/* CPU port of CuMesh's parallel midpoint QEM collapse scheduling. */
#include "mesh.hh"
#include <numeric>
#include <parallel/algorithm>
namespace px {
using V3 = lightrt::Vec3;
static V3 vertex(const Mesh &m, int i) {
    return V3(m.v[3 * i], m.v[3 * i + 1], m.v[3 * i + 2]);
}
static uint64_t edge(int a, int b) {
    if (a > b)
        std::swap(a, b);
    return (uint64_t(a) << 32) | uint32_t(b);
}
void simplify(Mesh &m, int target) {
    float threshold = 1e-8f;
    int stalled = 0;
    while (m.numF() > uint32_t(target)) {
        int nv = int(m.numV()), nf = int(m.numF());
        // Compact adjacency avoids millions of small allocations each round.
        // Filling it in face order preserves the former traversal order.
        std::vector<int> offsets(size_t(nv) + 1), adjacent(size_t(nf) * 3);
        std::vector<uint64_t> edges;
        edges.reserve(m.f.size());
        std::vector<std::array<float, 10>> qem(nv);
        for (int f = 0; f < nf; ++f) {
            int a = m.f[3 * f], b = m.f[3 * f + 1], c = m.f[3 * f + 2];
            ++offsets[size_t(a) + 1];
            ++offsets[size_t(b) + 1];
            ++offsets[size_t(c) + 1];
            edges.push_back(edge(a, b));
            edges.push_back(edge(b, c));
            edges.push_back(edge(c, a));
            V3 n = (vertex(m, b) - vertex(m, a)).cross(vertex(m, c) - vertex(m, a));
            float len = std::sqrt(n.dot(n));
            if (len > 1e-12f)
                n = n * (1 / len);
            else
                n = V3(0, 0, 0);
            float p[4] = {n.x, n.y, n.z, -n.dot(vertex(m, a))};
            for (int id : {a, b, c}) {
                int k = 0;
                for (int i = 0; i < 4; ++i)
                    for (int j = i; j < 4; ++j)
                        qem[id][k++] += p[i] * p[j];
            }
        }
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
        std::vector<int> cursor(offsets.begin(), offsets.end() - 1);
        for (int f = 0; f < nf; ++f)
            for (int j = 0; j < 3; ++j)
                adjacent[cursor[m.f[3 * f + j]]++] = f;
        __gnu_parallel::sort(edges.begin(), edges.end());
        std::vector<uint8_t> boundary(nv);
        for (size_t i = 0; i < edges.size();) {
            size_t j = i + 1;
            while (j < edges.size() && edges[j] == edges[i])
                ++j;
            if (j == i + 1) {
                boundary[int(edges[i] >> 32)] = 1;
                boundary[uint32_t(edges[i])] = 1;
            }
            i = j;
        }
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        std::vector<float> costs(edges.size());
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < edges.size(); ++i) {
            int a = int(edges[i] >> 32), b = uint32_t(edges[i]);
            V3 v0 = vertex(m, a), v1 = vertex(m, b);
            float w = boundary[a] && !boundary[b] ? 1 : !boundary[a] && boundary[b] ? 0 : .5f;
            V3 v = v0 * w + v1 * (1 - w);
            float p[4] = {v.x, v.y, v.z, 1}, cost = 0;
            int k = 0;
            for (int x = 0; x < 4; ++x)
                for (int y = x; y < 4; ++y) {
                    cost += (x == y ? 1 : 2) * (qem[a][k] + qem[b][k]) * p[x] * p[y];
                    ++k;
                }
            float length = (v1 - v0).dot(v1 - v0), skinny = 0;
            int triangles = 0;
            bool valid = true;
            for (int endpoint : {a, b})
                for (int position = offsets[endpoint]; position < offsets[endpoint + 1]; ++position) {
                    int f = adjacent[position];
                    int ids[3] = {m.f[3 * f], m.f[3 * f + 1], m.f[3 * f + 2]}, other = endpoint == a ? b : a;
                    if (ids[0] == other || ids[1] == other || ids[2] == other)
                        continue;
                    V3 old[3] = {vertex(m, ids[0]), vertex(m, ids[1]), vertex(m, ids[2])},
                       next[3] = {old[0], old[1], old[2]};
                    for (int j = 0; j < 3; ++j)
                        if (ids[j] == endpoint)
                            next[j] = v;
                    V3 old_n = (old[1] - old[0]).cross(old[2] - old[0]),
                       n = (next[1] - next[0]).cross(next[2] - next[0]);
                    if (old_n.dot(n) < 0) {
                        valid = false;
                        break;
                    }
                    float denominator = 0;
                    for (int j = 0; j < 3; ++j) {
                        V3 d = next[(j + 1) % 3] - next[j];
                        denominator += d.dot(d);
                    }
                    float metric = 2 * std::sqrt(3.f) * std::sqrt(n.dot(n)) / std::max(denominator, 1e-12f);
                    skinny += 1 - std::clamp(metric, 0.f, 1.f);
                    ++triangles;
                }
            costs[i] = valid ? cost + .01f * length + .001f * (triangles ? skinny / triangles : 0) * length
                             : INFINITY;
        }
        auto packed = [&](size_t i) {
            uint32_t u;
            std::memcpy(&u, &costs[i], 4);
            return (uint64_t(u) << 32) | uint32_t(i);
        };
        std::vector<uint64_t> best(nf, UINT64_MAX);
        auto atomic_min = [](uint64_t *dst, uint64_t value) {
            uint64_t old = __atomic_load_n(dst, __ATOMIC_RELAXED);
            while (value < old &&
                   !__atomic_compare_exchange_n(dst, &old, value, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            }
        };
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < edges.size(); ++i) {
            uint64_t p = packed(i);
            int a = int(edges[i] >> 32), b = uint32_t(edges[i]);
            for (int position = offsets[a]; position < offsets[a + 1]; ++position)
                atomic_min(&best[adjacent[position]], p);
            for (int position = offsets[b]; position < offsets[b + 1]; ++position)
                atomic_min(&best[adjacent[position]], p);
        }
        std::vector<int> mapping(nv);
        std::iota(mapping.begin(), mapping.end(), 0);
        std::vector<uint8_t> keep(nf, 1);
        int collapsed = 0;
        for (size_t i = 0; i < edges.size(); ++i) {
            if (costs[i] > threshold)
                continue;
            uint64_t p = packed(i);
            int a = int(edges[i] >> 32), b = uint32_t(edges[i]);
            bool valid = true;
            for (int position = offsets[a]; position < offsets[a + 1]; ++position)
                if (best[adjacent[position]] != p) {
                    valid = false;
                    break;
                }
            for (int position = offsets[b]; position < offsets[b + 1]; ++position)
                if (best[adjacent[position]] != p) {
                    valid = false;
                    break;
                }
            if (!valid)
                continue;
            float w = boundary[a] && !boundary[b] ? 1 : !boundary[a] && boundary[b] ? 0 : .5f;
            V3 v = vertex(m, a) * w + vertex(m, b) * (1 - w);
            m.v[3 * a] = v.x;
            m.v[3 * a + 1] = v.y;
            m.v[3 * a + 2] = v.z;
            mapping[b] = a;
            ++collapsed;
            for (int position = offsets[a]; position < offsets[a + 1]; ++position) {
                int f = adjacent[position];
                if (m.f[3 * f] == b || m.f[3 * f + 1] == b || m.f[3 * f + 2] == b)
                    keep[f] = 0;
            }
        }
        Mesh next;
        std::vector<int> compact(nv, -1);
        for (int i = 0; i < nv; ++i)
            if (mapping[i] == i) {
                compact[i] = int(next.numV());
                next.v.insert(next.v.end(), m.v.begin() + 3 * i, m.v.begin() + 3 * i + 3);
            }
        for (int f = 0; f < nf; ++f)
            if (keep[f])
                for (int j = 0; j < 3; ++j)
                    next.f.push_back(compact[mapping[m.f[3 * f + j]]]);
        float removed = float(nf - next.numF()) / nf;
        m = std::move(next);
        if (removed < .01f)
            threshold *= 10;
        stalled = collapsed ? 0 : stalled + 1;
        require(stalled < 20, "Mesh cannot be simplified to requested target");
        std::fprintf(stderr, "Pixal3D simplify: %u faces\n", m.numF());
    }
}
} // namespace px
