/* GPU port of the normal-cone chart merge in cpu/pixal3d/unwrap.cc (CuMesh
 * uv_unwrap's clustering). Every float sum keeps the CPU order: chart sums
 * run over faces in face order, pair lengths in adjacency order, perimeters
 * in chart-pair order (stable radix sorts group them), and acosf is glibc
 * 2.39's (fdlibm) algorithm, checked equal on every float in [-1, 1]. The
 * charts therefore match the CPU merge exactly on that glibc. CUDA only. */

namespace px_charts {
using namespace px_remesh;

// glibc 2.39 __ieee754_acosf (fdlibm e_acosf.c), float operations only.
__device__ float glibc_acosf(float x) {
    const float one = 1.0000000000e+00f, pi = 3.1415925026e+00f, pio2_hi = 1.5707962513e+00f,
                pio2_lo = 7.5497894159e-08f, pS0 = 1.6666667163e-01f, pS1 = -3.2556581497e-01f,
                pS2 = 2.0121252537e-01f, pS3 = -4.0055535734e-02f, pS4 = 7.9153501429e-04f,
                pS5 = 3.4793309169e-05f, qS1 = -2.4033949375e+00f, qS2 = 2.0209457874e+00f,
                qS3 = -6.8828397989e-01f, qS4 = 7.7038154006e-02f;
    float z, p, q, r, w, s, c, df;
    int hx = __float_as_int(x), ix = hx & 0x7fffffff;
    if (ix == 0x3f800000)
        return hx > 0 ? 0.0f : pi + 2.0f * pio2_lo;
    if (ix > 0x3f800000)
        return (x - x) / (x - x);
    if (ix < 0x3f000000) {
        if (ix <= 0x32800000)
            return pio2_hi + pio2_lo;
        z = x * x;
        p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))));
        q = one + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
        r = p / q;
        return pio2_hi - (x - (pio2_lo - x * r));
    } else if (hx < 0) {
        z = (one + x) * 0.5f;
        p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))));
        q = one + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
        s = sqrtf(z);
        r = p / q;
        w = r * s - pio2_lo;
        return pi - 2.0f * (s + w);
    }
    z = (one - x) * 0.5f;
    s = sqrtf(z);
    df = __int_as_float(__float_as_int(s) & 0xfffff000);
    c = (z - df * df) / (s + df);
    p = z * (pS0 + z * (pS1 + z * (pS2 + z * (pS3 + z * (pS4 + z * pS5)))));
    q = one + z * (qS1 + z * (qS2 + z * (qS3 + z * qS4)));
    r = p / q;
    w = r * s + c;
    return 2.0f * (df + w);
}
// std::clamp(v, -1, 1), std::max and std::min as the CPU evaluates them.
__device__ __forceinline__ float clamp_unit(float v) { return v < -1.f ? -1.f : 1.f < v ? 1.f : v; }
__device__ __forceinline__ float std_max(float a, float b) { return a < b ? b : a; }
__device__ __forceinline__ float std_min(float a, float b) { return b < a ? b : a; }

__global__ void iota_faces(int *x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = i;
}
__global__ void count_keys(int *counts, const int *keys, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        atomicAdd(counts + keys[i], 1);
}
// Axis, area and cone angle of each chart, over its faces in face order.
__global__ void chart_stats(float3 *axes, float *area, float *angle, int *size, const int *offsets, const int *faces,
                            const float *normals, const float *areas, int nc) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc)
        return;
    size[c] = offsets[c + 1] - offsets[c];
    float ax = 0, ay = 0, az = 0, sum = 0;
    for (int k = offsets[c]; k < offsets[c + 1]; ++k) {
        int f = faces[k];
        ax = ax + normals[3 * f];
        ay = ay + normals[3 * f + 1];
        az = az + normals[3 * f + 2];
        sum += areas[f];
    }
    float scale = 1 / std_max(sqrtf(ax * ax + ay * ay + az * az), 1e-20f);
    ax = ax * scale;
    ay = ay * scale;
    az = az * scale;
    float cone = 0;
    for (int k = offsets[c]; k < offsets[c + 1]; ++k) {
        int f = faces[k];
        float d = ax * normals[3 * f] + ay * normals[3 * f + 1] + az * normals[3 * f + 2];
        cone = std_max(cone, glibc_acosf(clamp_unit(d)));
    }
    axes[c] = make_float3(ax, ay, az);
    area[c] = sum;
    angle[c] = cone;
}
// Chart pair of each face adjacency; same-chart adjacencies sort last.
__global__ void pair_keys(unsigned long long *keys, const int *charts, const int2 *adjacent, int na) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= na)
        return;
    int a = charts[adjacent[i].x], b = charts[adjacent[i].y];
    if (a == b) {
        keys[i] = ~0ull;
        return;
    }
    if (a > b) {
        int t = a;
        a = b;
        b = t;
    }
    keys[i] = ((unsigned long long)a << 32) | unsigned(b);
}
__global__ void run_starts(int *flags, const unsigned long long *keys, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        flags[i] = i == 0 || keys[i] != keys[i - 1];
}
// One chart pair per run of equal keys; lengths summed in adjacency order.
__global__ void pair_lengths(int2 *pairs, float *lengths, const unsigned long long *keys,
                             const float *sorted_lengths, const int *flags, const int *run, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !flags[i] || keys[i] == ~0ull) // same-chart adjacencies are dropped
        return;
    float length = sorted_lengths[i];
    int j = i + 1;
    while (j < n && keys[j] == keys[i])
        length += sorted_lengths[j++];
    int r = run[i];
    pairs[r] = make_int2(int(keys[i] >> 32), int(unsigned(keys[i])));
    lengths[r] = length;
}
__global__ void incidences(int *chart, int *pair, const int2 *pairs, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    chart[2 * i] = pairs[i].x;
    pair[2 * i] = i;
    chart[2 * i + 1] = pairs[i].y;
    pair[2 * i + 1] = i;
}
__global__ void perimeters(float *perimeter, const int *offsets, const int *pair_of, const float *lengths, int nc) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nc)
        return;
    float sum = 0;
    for (int k = offsets[c]; k < offsets[c + 1]; ++k)
        sum += lengths[pair_of[k]];
    perimeter[c] = sum;
}
__global__ void pair_costs(float *cost, unsigned long long *best, int *bad, const int2 *pairs, const float *lengths,
                           const float3 *axes, const float *area, const float *angle, const float *perimeter,
                           const int *size, int max_faces, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int a = pairs[i].x, b = pairs[i].y;
    float3 pa = axes[a], pb = axes[b];
    float axis_angle = glibc_acosf(clamp_unit(pa.x * pb.x + pa.y * pb.y + pa.z * pb.z));
    float low = std_min(-angle[a], axis_angle - angle[b]), high = std_max(angle[a], axis_angle + angle[b]);
    float ar = area[a] + area[b], per = perimeter[a] + perimeter[b] - 2 * lengths[i];
    float value = size[a] + size[b] > max_faces
                      ? INFINITY
                      : (high - low) * .5f + .1f * ar + .0001f * per * per / std_max(ar, 1e-20f);
    cost[i] = value;
    // The packed minimum is the CPU's first strict minimum for non-negative
    // costs (+inf included); anything else goes back to the CPU.
    if (!(value >= 0.f) || __float_as_uint(value) == 0x80000000u)
        atomicExch(bad, 1);
    unsigned long long packed = ((unsigned long long)__float_as_uint(value) << 32) | unsigned(i);
    atomicMin(best + a, packed);
    atomicMin(best + b, packed);
}
// A winner is the best pair of both charts, so winners share no chart.
__global__ void winners(int *map, int *count, const float *cost, const unsigned long long *best, const int2 *pairs,
                        int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !(cost[i] <= 1.57079632679f))
        return;
    int a = pairs[i].x, b = pairs[i].y;
    if (unsigned(best[a]) == unsigned(i) && unsigned(best[b]) == unsigned(i)) {
        map[b] = a;
        atomicAdd(count, 1);
    }
}
__global__ void roots(int *flag, const int *map, int n) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < n)
        flag[c] = map[c] == c;
}
__global__ void relabel(int *charts, const int *map, const int *compact, int nf) {
    int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < nf)
        charts[f] = compact[map[charts[f]]];
}

// Faces (or pairs) grouped by `keys` in [0, n), each group in input order.
inline void group(Arena &arena, const int *keys, int count, int n, int *offsets, int *members, int *member_input) {
    int *counts = arena.alloc<int>(std::max(n, 1));
    RM_CHECK(cudaMemsetAsync(counts, 0, size_t(n) * 4, arena.stream));
    count_keys<<<blocks(count), 256, 0, arena.stream>>>(counts, keys, count);
    int total = scan(arena, counts, offsets, n);
    RM_CHECK(cudaMemcpyAsync(offsets + n, &total, 4, cudaMemcpyHostToDevice, arena.stream));
    arena.release(counts);
    int *sorted_keys = arena.alloc<int>(count);
    size_t bytes = 0;
    int bits = 1;
    while (bits < 31 && (1 << bits) < n)
        ++bits;
    RM_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, sorted_keys, member_input, members, count, 0, bits,
                                             arena.stream));
    void *temp = arena.alloc<char>(bytes);
    RM_CHECK(cub::DeviceRadixSort::SortPairs(temp, bytes, keys, sorted_keys, member_input, members, count, 0, bits,
                                             arena.stream));
    arena.release(temp);
    arena.release(sorted_keys);
}

inline void run(const px_chart_request &r, px_chart_result &result) {
    int nf = r.num_faces, na = r.num_adjacent;
    if (nf < 1)
        throw Error("chart merge needs faces");
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
    float *normals = arena.alloc<float>(size_t(nf) * 3), *areas = arena.alloc<float>(nf);
    int2 *adjacent = arena.alloc<int2>(std::max(na, 1));
    float *adjacent_length = arena.alloc<float>(std::max(na, 1));
    RM_CHECK(cudaMemcpyAsync(normals, r.normals, size_t(nf) * 12, cudaMemcpyHostToDevice, s));
    RM_CHECK(cudaMemcpyAsync(areas, r.areas, size_t(nf) * 4, cudaMemcpyHostToDevice, s));
    RM_CHECK(cudaMemcpyAsync(adjacent, r.adjacent_faces, size_t(na) * 8, cudaMemcpyHostToDevice, s));
    RM_CHECK(cudaMemcpyAsync(adjacent_length, r.adjacent_lengths, size_t(na) * 4, cudaMemcpyHostToDevice, s));
    int *charts = arena.alloc<int>(nf), *face_ids = arena.alloc<int>(nf);
    iota_faces<<<blocks(nf), 256, 0, s>>>(charts, nf);
    iota_faces<<<blocks(nf), 256, 0, s>>>(face_ids, nf);
    int *bad = arena.alloc<int>(1);
    RM_CHECK(cudaMemsetAsync(bad, 0, 4, s));
    int nc = nf, rounds = 0;
    for (;;) {
        ++rounds;
        // Chart axis, area and cone angle.
        int *chart_offsets = arena.alloc<int>(size_t(nc) + 1), *chart_faces = arena.alloc<int>(nf);
        group(arena, charts, nf, nc, chart_offsets, chart_faces, face_ids);
        auto *axes = arena.alloc<float3>(nc);
        float *area = arena.alloc<float>(nc), *angle = arena.alloc<float>(nc), *perimeter = arena.alloc<float>(nc);
        int *size = arena.alloc<int>(nc);
        chart_stats<<<blocks(nc), 256, 0, s>>>(axes, area, angle, size, chart_offsets, chart_faces, normals, areas, nc);
        RM_CHECK(cudaGetLastError());
        arena.release(chart_offsets);
        arena.release(chart_faces);
        // Chart pairs: adjacency lengths grouped by pair, in adjacency order.
        int np = 0;
        int2 *pairs = nullptr;
        float *lengths = nullptr;
        if (na > 0) {
            auto *keys = arena.alloc<unsigned long long>(na), *sorted = arena.alloc<unsigned long long>(na);
            float *sorted_lengths = arena.alloc<float>(na);
            pair_keys<<<blocks(na), 256, 0, s>>>(keys, charts, adjacent, na);
            size_t bytes = 0;
            RM_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, bytes, keys, sorted, adjacent_length, sorted_lengths, na,
                                                     0, 64, s));
            void *temp = arena.alloc<char>(bytes);
            RM_CHECK(cub::DeviceRadixSort::SortPairs(temp, bytes, keys, sorted, adjacent_length, sorted_lengths, na, 0,
                                                     64, s));
            arena.release(temp);
            arena.release(keys);
            int *flags = arena.alloc<int>(na), *run_index = arena.alloc<int>(na);
            run_starts<<<blocks(na), 256, 0, s>>>(flags, sorted, na);
            int runs = scan(arena, flags, run_index, na);
            // The ~0 run (same-chart adjacencies) is last; drop it.
            unsigned long long last = arena.scalar(sorted + na - 1);
            np = last == ~0ull ? runs - 1 : runs;
            pairs = arena.alloc<int2>(std::max(runs, 1));
            lengths = arena.alloc<float>(std::max(runs, 1));
            pair_lengths<<<blocks(na), 256, 0, s>>>(pairs, lengths, sorted, sorted_lengths, flags, run_index, na);
            RM_CHECK(cudaGetLastError());
            for (void *p : {(void *)sorted, (void *)sorted_lengths, (void *)flags, (void *)run_index})
                arena.release(p);
        }
        if (np > 0) {
            // Perimeters: each chart's pair lengths in pair order.
            int *incident_chart = arena.alloc<int>(size_t(np) * 2), *incident_pair = arena.alloc<int>(size_t(np) * 2);
            incidences<<<blocks(np), 256, 0, s>>>(incident_chart, incident_pair, pairs, np);
            int *pair_offsets = arena.alloc<int>(size_t(nc) + 1), *pair_of = arena.alloc<int>(size_t(np) * 2);
            group(arena, incident_chart, np * 2, nc, pair_offsets, pair_of, incident_pair);
            perimeters<<<blocks(nc), 256, 0, s>>>(perimeter, pair_offsets, pair_of, lengths, nc);
            for (void *p : {(void *)incident_chart, (void *)incident_pair, (void *)pair_offsets, (void *)pair_of})
                arena.release(p);
        }
        int merged = 0;
        int *map = arena.alloc<int>(nc);
        iota_faces<<<blocks(nc), 256, 0, s>>>(map, nc);
        if (np > 0) {
            float *cost = arena.alloc<float>(np);
            auto *best = arena.alloc<unsigned long long>(nc);
            int *count = arena.alloc<int>(1);
            RM_CHECK(cudaMemsetAsync(best, 0xff, size_t(nc) * 8, s));
            RM_CHECK(cudaMemsetAsync(count, 0, 4, s));
            pair_costs<<<blocks(np), 256, 0, s>>>(cost, best, bad, pairs, lengths, axes, area, angle, perimeter, size,
                                                  r.max_chart_faces > 0 ? r.max_chart_faces : INT_MAX, np);
            winners<<<blocks(np), 256, 0, s>>>(map, count, cost, best, pairs, np);
            RM_CHECK(cudaGetLastError());
            merged = arena.scalar(count);
            if (arena.scalar(bad))
                throw Error("chart merge met a cost outside the GPU's exact range");
            for (void *p : {(void *)cost, (void *)best, (void *)count})
                arena.release(p);
        }
        for (void *p : {(void *)axes, (void *)area, (void *)angle, (void *)perimeter, (void *)size, (void *)pairs,
                        (void *)lengths})
            if (p)
                arena.release(p);
        if (!merged) {
            arena.release(map);
            break;
        }
        int *flag = arena.alloc<int>(nc), *compact = arena.alloc<int>(nc);
        roots<<<blocks(nc), 256, 0, s>>>(flag, map, nc);
        int next = scan(arena, flag, compact, nc);
        relabel<<<blocks(nf), 256, 0, s>>>(charts, map, compact, nf);
        RM_CHECK(cudaGetLastError());
        for (void *p : {(void *)flag, (void *)compact, (void *)map})
            arena.release(p);
        nc = next;
    }
    result.charts = static_cast<int32_t *>(std::malloc(size_t(nf) * 4));
    if (!result.charts)
        throw Error("Cannot allocate the chart map");
    RM_CHECK(cudaMemcpyAsync(result.charts, charts, size_t(nf) * 4, cudaMemcpyDeviceToHost, s));
    RM_CHECK(cudaStreamSynchronize(s));
    result.num_charts = nc;
    result.rounds = rounds;
}
} // namespace px_charts

extern "C" int px_gpu_merge_charts(const px_chart_request *request, px_chart_result *result) {
    *result = px_chart_result{};
    try {
        px_charts::run(*request, *result);
        return 0;
    } catch (const std::exception &e) {
        std::free(result->charts);
        result->charts = nullptr;
        std::snprintf(result->error, sizeof(result->error), "%s", e.what());
        return -1;
    }
}
