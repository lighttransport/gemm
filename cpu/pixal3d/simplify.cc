/* CPU port of CuMesh's parallel midpoint QEM collapse scheduling. */
#include "mesh.hh"
#include <chrono>
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
    int stalled = 0, rounds = 0;
    // Per-phase totals, printed once: where the rounds spend their time.
    using clock = std::chrono::steady_clock;
    double spent[8] = {};
    auto tick = clock::now();
    auto lap = [&](int phase) {
        auto now = clock::now();
        spent[phase] += std::chrono::duration<double>(now - tick).count();
        tick = now;
    };
    auto input_faces = m.numF();
    while (m.numF() > uint32_t(target)) {
        ++rounds;
        tick = clock::now();
        int nv = int(m.numV()), nf = int(m.numF());
        // Compact adjacency avoids millions of small allocations each round.
        // Filling it in face order preserves the former traversal order.
        // Every step below is parallel yet bitwise identical to the serial
        // form: integer counts commute; each vertex's quadric is summed over
        // its faces in face order (the CSR list is filled in face order),
        // exactly the order of the former single face loop.
        std::vector<int> offsets(size_t(nv) + 1), adjacent(size_t(nf) * 3);
        std::vector<uint64_t> edges(size_t(nf) * 3);
        std::vector<std::array<float, 4>> planes(nf);
#pragma omp parallel for schedule(static)
        for (int f = 0; f < nf; ++f) {
            int a = m.f[3 * f], b = m.f[3 * f + 1], c = m.f[3 * f + 2];
            __atomic_fetch_add(&offsets[size_t(a) + 1], 1, __ATOMIC_RELAXED);
            __atomic_fetch_add(&offsets[size_t(b) + 1], 1, __ATOMIC_RELAXED);
            __atomic_fetch_add(&offsets[size_t(c) + 1], 1, __ATOMIC_RELAXED);
            edges[3 * size_t(f)] = edge(a, b);
            edges[3 * size_t(f) + 1] = edge(b, c);
            edges[3 * size_t(f) + 2] = edge(c, a);
            V3 n = (vertex(m, b) - vertex(m, a)).cross(vertex(m, c) - vertex(m, a));
            float len = std::sqrt(n.dot(n));
            if (len > 1e-12f)
                n = n * (1 / len);
            else
                n = V3(0, 0, 0);
            planes[f] = {n.x, n.y, n.z, -n.dot(vertex(m, a))};
        }
        lap(0);
        std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
        {
            std::vector<int> cursor(offsets.begin(), offsets.end() - 1);
            for (int f = 0; f < nf; ++f)
                for (int j = 0; j < 3; ++j)
                    adjacent[cursor[m.f[3 * f + j]]++] = f;
        }
        lap(1);
        std::vector<std::array<float, 10>> qem(nv);
#pragma omp parallel for schedule(static)
        for (int v = 0; v < nv; ++v)
            for (int position = offsets[v]; position < offsets[v + 1]; ++position) {
                const float *p = planes[adjacent[position]].data();
                int k = 0;
                for (int i = 0; i < 4; ++i)
                    for (int j = i; j < 4; ++j)
                        qem[v][k++] += p[i] * p[j];
            }
        planes.clear();
        planes.shrink_to_fit();
        lap(2);
        __gnu_parallel::sort(edges.begin(), edges.end());
        std::vector<uint8_t> boundary(nv);
        size_t ne = edges.size();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < ne; ++i) {
            // A boundary edge appears once: it starts a run of length one.
            bool first = i == 0 || edges[i - 1] != edges[i];
            bool last = i + 1 == ne || edges[i + 1] != edges[i];
            if (first && last) {
                __atomic_store_n(&boundary[int(edges[i] >> 32)], uint8_t(1), __ATOMIC_RELAXED);
                __atomic_store_n(&boundary[uint32_t(edges[i])], uint8_t(1), __ATOMIC_RELAXED);
            }
        }
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        lap(3);
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
        lap(4);
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
        lap(5);
        std::vector<int> mapping(nv);
        std::iota(mapping.begin(), mapping.end(), 0);
        std::vector<uint8_t> keep(nf, 1);
        int collapsed = 0;
        // Winners are disjoint: a winning edge is the best of every face
        // around both endpoints, so two winners share no face and no vertex.
        // Applying them in parallel writes disjoint entries.
#pragma omp parallel for schedule(static) reduction(+ : collapsed)
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
        lap(6);
        // Compaction by prefix sums, in the original order.
        Mesh next;
        std::vector<int> compact(nv, -1);
        {
            std::vector<int> vpos(size_t(nv) + 1, 0), fpos(size_t(nf) + 1, 0);
            for (int i = 0; i < nv; ++i)
                vpos[i + 1] = vpos[i] + (mapping[i] == i);
            for (int f = 0; f < nf; ++f)
                fpos[f + 1] = fpos[f] + keep[f];
            next.v.resize(size_t(vpos[nv]) * 3);
            next.f.resize(size_t(fpos[nf]) * 3);
#pragma omp parallel for schedule(static)
            for (int i = 0; i < nv; ++i)
                if (mapping[i] == i) {
                    compact[i] = vpos[i];
                    std::copy_n(m.v.begin() + 3 * size_t(i), 3, next.v.begin() + 3 * size_t(vpos[i]));
                }
#pragma omp parallel for schedule(static)
            for (int f = 0; f < nf; ++f)
                if (keep[f])
                    for (int j = 0; j < 3; ++j)
                        next.f[3 * size_t(fpos[f]) + j] = compact[mapping[m.f[3 * f + j]]];
        }
        lap(7);
        float removed = float(nf - next.numF()) / nf;
        m = std::move(next);
        if (removed < .01f)
            threshold *= 10;
        stalled = collapsed ? 0 : stalled + 1;
        require(stalled < 20, "Mesh cannot be simplified to requested target");
    }
    std::fprintf(stderr,
                 "Pixal3D simplify: %u -> %u faces in %d rounds (setup %.2f, adjacency %.2f, qem %.2f, sort+boundary "
                 "%.2f, costs %.2f, select %.2f, collapse %.2f, compact %.2f s)\n",
                 input_faces, m.numF(), rounds, spent[0], spent[1], spent[2], spent[3], spent[4], spent[5], spent[6],
                 spent[7]);
}
bool simplify_gpu(const GpuGeometry &gpu, Mesh &mesh, int target) {
    if (!gpu.simplify || mesh.numF() <= uint32_t(target))
        return false;
    px_simplify_request request{mesh.v.data(), mesh.f.data(), int(mesh.numV()), int(mesh.numF()), target, 0, gpu.gpu};
    px_remesh_result result{};
    if (gpu.simplify(&request, &result) != 0) {
        std::fprintf(stderr, "Pixal3D simplify: GPU path unavailable (%s); using the CPU\n", result.error);
        return false;
    }
    mesh.set(result.vertices, result.num_vertices, result.faces, result.num_faces);
    std::free(result.vertices);
    std::free(result.faces);
    return true;
}
} // namespace px
