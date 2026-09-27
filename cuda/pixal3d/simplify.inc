/* GPU port of px::simplify (cpu/pixal3d/simplify.cc): CuMesh's parallel
 * midpoint QEM collapse rounds. Each CPU step is already order-independent
 * or ordered (CSR in face order, per-vertex quadric sums, packed atomicMin
 * selection, scan compaction), and every float expression keeps the CPU's
 * order (--fmad=false), so the result is bit-identical. Uses the remesh's
 * Arena: allocations borrow from the engine's budget. CUDA only. */

namespace px_simplify {
using namespace px_remesh;

struct F3 {
    float x, y, z;
};
__device__ __forceinline__ F3 load(const float *v, int i) { return {v[3 * i], v[3 * i + 1], v[3 * i + 2]}; }
__device__ __forceinline__ F3 operator+(F3 a, F3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__device__ __forceinline__ F3 operator-(F3 a, F3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__device__ __forceinline__ F3 operator*(F3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
__device__ __forceinline__ float dot(F3 a, F3 v) { return a.x * v.x + a.y * v.y + a.z * v.z; }
__device__ __forceinline__ F3 cross(F3 a, F3 v) {
    return {a.y * v.z - a.z * v.y, a.z * v.x - a.x * v.z, a.x * v.y - a.y * v.x};
}
__device__ __forceinline__ unsigned long long edge_key(int a, int b) {
    if (a > b) {
        int t = a;
        a = b;
        b = t;
    }
    return ((unsigned long long)a << 32) | unsigned(b);
}
__device__ __forceinline__ float midpoint_weight(const unsigned char *boundary, int a, int b) {
    return boundary[a] && !boundary[b] ? 1 : !boundary[a] && boundary[b] ? 0 : .5f;
}

__global__ void setup(float4 *planes, unsigned long long *edges, int *counts, int *corner_vertex, int *corner_face,
                      const float *V, const int *F, int nf) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nf)
        return;
    int a = F[3 * f], b = F[3 * f + 1], c = F[3 * f + 2];
    atomicAdd(counts + a, 1);
    atomicAdd(counts + b, 1);
    atomicAdd(counts + c, 1);
    edges[3 * size_t(f)] = edge_key(a, b);
    edges[3 * size_t(f) + 1] = edge_key(b, c);
    edges[3 * size_t(f) + 2] = edge_key(c, a);
    for (int j = 0; j < 3; ++j) {
        corner_vertex[3 * size_t(f) + j] = F[3 * f + j];
        corner_face[3 * size_t(f) + j] = f;
    }
    F3 va = load(V, a);
    F3 n = cross(load(V, b) - va, load(V, c) - va);
    float len = sqrtf(dot(n, n));
    if (len > 1e-12f)
        n = n * (1 / len);
    else
        n = F3{0, 0, 0};
    planes[f] = make_float4(n.x, n.y, n.z, -dot(n, va));
}
// Each vertex's quadric, summed over its faces in face order (the CSR order).
__global__ void quadrics(float *qem, const int *offsets, const int *adjacent, const float4 *planes, int nv) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= nv)
        return;
    float q[10] = {};
    for (int position = offsets[v]; position < offsets[v + 1]; ++position) {
        float4 plane = planes[adjacent[position]];
        float p[4] = {plane.x, plane.y, plane.z, plane.w};
        int k = 0;
        for (int i = 0; i < 4; ++i)
            for (int j = i; j < 4; ++j)
                q[k++] += p[i] * p[j];
    }
    for (int k = 0; k < 10; ++k)
        qem[size_t(v) * 10 + k] = q[k];
}
// A boundary edge appears once in the sorted list.
__global__ void boundaries(unsigned char *boundary, const unsigned long long *edges, int ne) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne)
        return;
    bool first = i == 0 || edges[i - 1] != edges[i];
    bool last = i + 1 == ne || edges[i + 1] != edges[i];
    if (first && last) {
        boundary[int(edges[i] >> 32)] = 1;
        boundary[unsigned(edges[i])] = 1;
    }
}
__global__ void edge_costs(float *costs, const unsigned long long *edges, int ne, const float *V, const int *F,
                           const float *qem, const unsigned char *boundary, const int *offsets, const int *adjacent) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne)
        return;
    int a = int(edges[i] >> 32), b = int(unsigned(edges[i]));
    F3 v0 = load(V, a), v1 = load(V, b);
    float w = midpoint_weight(boundary, a, b);
    F3 v = v0 * w + v1 * (1 - w);
    float p[4] = {v.x, v.y, v.z, 1}, cost = 0;
    const float *qa = qem + size_t(a) * 10, *qb = qem + size_t(b) * 10;
    int k = 0;
    for (int x = 0; x < 4; ++x)
        for (int y = x; y < 4; ++y) {
            cost += (x == y ? 1 : 2) * (qa[k] + qb[k]) * p[x] * p[y];
            ++k;
        }
    float length = dot(v1 - v0, v1 - v0), skinny = 0;
    int triangles = 0;
    bool valid = true;
    const float sqrt3 = sqrtf(3.f);
    for (int e = 0; e < 2; ++e) {
        int endpoint = e ? b : a, other = e ? a : b;
        for (int position = offsets[endpoint]; position < offsets[endpoint + 1]; ++position) {
            int f = adjacent[position];
            int ids[3] = {F[3 * f], F[3 * f + 1], F[3 * f + 2]};
            if (ids[0] == other || ids[1] == other || ids[2] == other)
                continue;
            F3 old[3] = {load(V, ids[0]), load(V, ids[1]), load(V, ids[2])}, next[3] = {old[0], old[1], old[2]};
            for (int j = 0; j < 3; ++j)
                if (ids[j] == endpoint)
                    next[j] = v;
            F3 old_n = cross(old[1] - old[0], old[2] - old[0]), n = cross(next[1] - next[0], next[2] - next[0]);
            if (dot(old_n, n) < 0) {
                valid = false;
                break;
            }
            float denominator = 0;
            for (int j = 0; j < 3; ++j) {
                F3 d = next[(j + 1) % 3] - next[j];
                denominator += dot(d, d);
            }
            // std::max / std::clamp semantics, NaN included.
            float bounded = denominator < 1e-12f ? 1e-12f : denominator;
            float metric = 2 * sqrt3 * sqrtf(dot(n, n)) / bounded;
            float clamped = metric < 0.f ? 0.f : 1.f < metric ? 1.f : metric;
            skinny += 1 - clamped;
            ++triangles;
        }
    }
    costs[i] = valid ? cost + .01f * length + .001f * (triangles ? skinny / triangles : 0) * length : INFINITY;
}
__device__ __forceinline__ unsigned long long packed_cost(const float *costs, int i) {
    return ((unsigned long long)__float_as_uint(costs[i]) << 32) | unsigned(i);
}
__global__ void select_best(unsigned long long *best, const float *costs, const unsigned long long *edges, int ne,
                            const int *offsets, const int *adjacent) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne)
        return;
    unsigned long long p = packed_cost(costs, i);
    int a = int(edges[i] >> 32), b = int(unsigned(edges[i]));
    for (int position = offsets[a]; position < offsets[a + 1]; ++position)
        atomicMin(best + adjacent[position], p);
    for (int position = offsets[b]; position < offsets[b + 1]; ++position)
        atomicMin(best + adjacent[position], p);
}
// Winners are the best edge of every face around both endpoints, so they
// share no face and no vertex: their writes are disjoint.
__global__ void collapse(float *V, int *mapping, int *keep, int *collapsed, const float *costs,
                         const unsigned long long *edges, int ne, float threshold, const unsigned long long *best,
                         const unsigned char *boundary, const int *F, const int *offsets, const int *adjacent) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ne || costs[i] > threshold)
        return;
    unsigned long long p = packed_cost(costs, i);
    int a = int(edges[i] >> 32), b = int(unsigned(edges[i]));
    for (int position = offsets[a]; position < offsets[a + 1]; ++position)
        if (best[adjacent[position]] != p)
            return;
    for (int position = offsets[b]; position < offsets[b + 1]; ++position)
        if (best[adjacent[position]] != p)
            return;
    float w = midpoint_weight(boundary, a, b);
    F3 v = load(V, a) * w + load(V, b) * (1 - w);
    V[3 * a] = v.x;
    V[3 * a + 1] = v.y;
    V[3 * a + 2] = v.z;
    mapping[b] = a;
    atomicAdd(collapsed, 1);
    for (int position = offsets[a]; position < offsets[a + 1]; ++position) {
        int f = adjacent[position];
        if (F[3 * f] == b || F[3 * f + 1] == b || F[3 * f + 2] == b)
            keep[f] = 0;
    }
}
__global__ void iota(int *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = i;
}
__global__ void fill(int *x, int n, int value) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = value;
}
__global__ void survivors(int *flag, const int *mapping, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        flag[i] = mapping[i] == i;
}
__global__ void compact(float *V2, int *F2, const float *V, const int *F, const int *mapping, const int *alive,
                        const int *vpos, const int *keep, const int *fpos, int nv, int nf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nv && alive[i])
        for (int d = 0; d < 3; ++d)
            V2[3 * size_t(vpos[i]) + d] = V[3 * size_t(i) + d];
    if (i < nf && keep[i])
        for (int j = 0; j < 3; ++j)
            F2[3 * size_t(fpos[i]) + j] = vpos[mapping[F[3 * i + j]]];
}

inline int bits_for(int n) {
    int bits = 1;
    while (bits < 31 && (1 << bits) < n)
        ++bits;
    return bits;
}

inline void run(const px_simplify_request &r, px_remesh_result &result) {
    using clock = std::chrono::steady_clock;
    auto t0 = clock::now();
    int nv = r.num_vertices, nf = r.num_faces, target = r.target;
    if (nv < 3 || nf < 1)
        throw Error("simplify needs a mesh");
    Arena arena;
    arena.engine = static_cast<Device *>(r.gpu);
    RM_CHECK(cudaSetDevice(arena.engine ? arena.engine->ordinal : r.device));
    arena.open_stream();
    if (!arena.engine) {
        size_t free = 0, total = 0;
        RM_CHECK(cudaMemGetInfo(&free, &total));
        arena.limit = free > (size_t(256) << 20) ? free - (size_t(256) << 20) : 0;
    }
    cudaStream_t s = arena.stream;
    float *V = arena.alloc<float>(size_t(nv) * 3);
    int *F = arena.alloc<int>(size_t(nf) * 3);
    RM_CHECK(cudaMemcpyAsync(V, r.vertices, size_t(nv) * 12, cudaMemcpyHostToDevice, s));
    RM_CHECK(cudaMemcpyAsync(F, r.faces, size_t(nf) * 12, cudaMemcpyHostToDevice, s));
    float threshold = 1e-8f;
    int stalled = 0, rounds = 0, input_faces = nf;
    while (nf > target) {
        ++rounds;
        size_t corners = size_t(nf) * 3;
        // Adjacency: corners stably sorted by vertex keep face order.
        auto *planes = arena.alloc<float4>(nf);
        auto *edges = arena.alloc<unsigned long long>(corners);
        int *counts = arena.alloc<int>(nv), *offsets = arena.alloc<int>(size_t(nv) + 1);
        int *corner_vertex = arena.alloc<int>(corners), *corner_face = arena.alloc<int>(corners);
        RM_CHECK(cudaMemsetAsync(counts, 0, size_t(nv) * 4, s));
        setup<<<blocks(nf), 256, 0, s>>>(planes, edges, counts, corner_vertex, corner_face, V, F, nf);
        RM_CHECK(cudaGetLastError());
        int total = scan(arena, counts, offsets, nv);
        RM_CHECK(cudaMemcpyAsync(offsets + nv, &total, 4, cudaMemcpyHostToDevice, s));
        arena.release(counts);
        int *vertex_spare = arena.alloc<int>(corners), *face_spare = arena.alloc<int>(corners);
        cub::DoubleBuffer<int> vertex_keys(corner_vertex, vertex_spare), faces_of(corner_face, face_spare);
        size_t bytes = 0;
        int vbits = bits_for(nv);
        RM_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes, vertex_keys, faces_of, int(corners), 0, vbits, s));
        void *temp = arena.alloc<char>(bytes);
        RM_CHECK(cub::DeviceRadixSort::SortPairs(temp, bytes, vertex_keys, faces_of, int(corners), 0, vbits, s));
        arena.release(temp);
        int *adjacent = faces_of.Current();
        arena.release(faces_of.Alternate());
        arena.release(corner_vertex);
        arena.release(vertex_spare);
        float *qem = arena.alloc<float>(size_t(nv) * 10);
        quadrics<<<blocks(nv), 256, 0, s>>>(qem, offsets, adjacent, planes, nv);
        RM_CHECK(cudaGetLastError());
        arena.release(planes);
        // Edges: sort, mark boundaries, deduplicate.
        auto *edge_spare = arena.alloc<unsigned long long>(corners);
        cub::DoubleBuffer<unsigned long long> edge_keys(edges, edge_spare);
        RM_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, bytes, edge_keys, int(corners), 0, 32 + vbits, s));
        temp = arena.alloc<char>(bytes);
        RM_CHECK(cub::DeviceRadixSort::SortKeys(temp, bytes, edge_keys, int(corners), 0, 32 + vbits, s));
        arena.release(temp);
        // Sorted edges in one buffer, the unique ones into the other.
        unsigned long long *sorted_edges = edge_keys.Current();
        edges = edge_keys.Alternate();
        auto *boundary = arena.alloc<unsigned char>(nv);
        RM_CHECK(cudaMemsetAsync(boundary, 0, nv, s));
        boundaries<<<blocks(corners), 256, 0, s>>>(boundary, sorted_edges, int(corners));
        int *unique_count = arena.alloc<int>(1);
        RM_CHECK(cub::DeviceSelect::Unique(nullptr, bytes, sorted_edges, edges, unique_count, int(corners), s));
        temp = arena.alloc<char>(bytes);
        RM_CHECK(cub::DeviceSelect::Unique(temp, bytes, sorted_edges, edges, unique_count, int(corners), s));
        arena.release(temp);
        arena.release(sorted_edges);
        int ne = arena.scalar(unique_count);
        arena.release(unique_count);
        // Costs, per-face best edge, disjoint collapses.
        float *costs = arena.alloc<float>(ne);
        edge_costs<<<blocks(ne), 256, 0, s>>>(costs, edges, ne, V, F, qem, boundary, offsets, adjacent);
        RM_CHECK(cudaGetLastError());
        arena.release(qem);
        auto *best = arena.alloc<unsigned long long>(nf);
        RM_CHECK(cudaMemsetAsync(best, 0xff, size_t(nf) * 8, s));
        select_best<<<blocks(ne), 256, 0, s>>>(best, costs, edges, ne, offsets, adjacent);
        int *mapping = arena.alloc<int>(nv), *keep = arena.alloc<int>(nf), *collapsed = arena.alloc<int>(1);
        iota<<<blocks(nv), 256, 0, s>>>(mapping, nv);
        fill<<<blocks(nf), 256, 0, s>>>(keep, nf, 1);
        RM_CHECK(cudaMemsetAsync(collapsed, 0, 4, s));
        collapse<<<blocks(ne), 256, 0, s>>>(V, mapping, keep, collapsed, costs, edges, ne, threshold, best, boundary,
                                            F, offsets, adjacent);
        RM_CHECK(cudaGetLastError());
        int collapses = arena.scalar(collapsed);
        for (void *p : {(void *)best, (void *)costs, (void *)edges, (void *)boundary, (void *)offsets,
                        (void *)adjacent, (void *)collapsed})
            arena.release(p);
        // Compaction by prefix sums, in the original order.
        int *alive = arena.alloc<int>(nv), *vpos = arena.alloc<int>(nv), *fpos = arena.alloc<int>(nf);
        survivors<<<blocks(nv), 256, 0, s>>>(alive, mapping, nv);
        int nv2 = scan(arena, alive, vpos, nv), nf2 = scan(arena, keep, fpos, nf);
        float *V2 = arena.alloc<float>(size_t(nv2) * 3);
        int *F2 = arena.alloc<int>(size_t(nf2) * 3);
        compact<<<blocks(std::max(nv, nf)), 256, 0, s>>>(V2, F2, V, F, mapping, alive, vpos, keep, fpos, nv, nf);
        RM_CHECK(cudaGetLastError());
        for (void *p : {(void *)alive, (void *)vpos, (void *)fpos, (void *)mapping, (void *)keep, (void *)V, (void *)F})
            arena.release(p);
        V = V2;
        F = F2;
        float removed = float(nf - nf2) / nf;
        nv = nv2;
        nf = nf2;
        if (removed < .01f)
            threshold *= 10;
        stalled = collapses ? 0 : stalled + 1;
        if (stalled >= 20)
            throw Error("Mesh cannot be simplified to requested target");
    }
    result.vertices = static_cast<float *>(std::malloc(size_t(nv) * 12));
    result.faces = static_cast<int32_t *>(std::malloc(size_t(nf) * 12));
    if (!result.vertices || !result.faces)
        throw Error("Cannot allocate the simplified mesh");
    RM_CHECK(cudaMemcpyAsync(result.vertices, V, size_t(nv) * 12, cudaMemcpyDeviceToHost, s));
    RM_CHECK(cudaMemcpyAsync(result.faces, F, size_t(nf) * 12, cudaMemcpyDeviceToHost, s));
    RM_CHECK(cudaStreamSynchronize(s));
    result.num_vertices = nv;
    result.num_faces = nf;
    std::fprintf(stderr, "Pixal3D simplify (GPU): %d -> %d faces in %d rounds, %.2fs, peak %.0f MiB\n", input_faces,
                 nf, rounds, std::chrono::duration<double>(clock::now() - t0).count(), arena.peak / 1048576.0);
}
} // namespace px_simplify

extern "C" int px_gpu_simplify(const px_simplify_request *request, px_remesh_result *result) {
    *result = px_remesh_result{};
    try {
        px_simplify::run(*request, *result);
        return 0;
    } catch (const std::exception &e) {
        std::free(result->vertices);
        std::free(result->faces);
        result->vertices = nullptr;
        result->faces = nullptr;
        std::snprintf(result->error, sizeof(result->error), "%s", e.what());
        return -1;
    }
}
