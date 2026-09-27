/* GPU port of px::remesh (cpu/pixal3d/remesh.cc): the same narrow-band UDF
 * levels and dual contouring, with an LBVH for the closest-point distances.
 * Distances follow the CPU expression order (built with --fmad=false) and
 * every list keeps the CPU order, so the mesh matches the CPU remesh. Runs on
 * its own stream with stream-ordered allocations, beside the resident engine;
 * it never touches the engine's Device. CUDA only. */
#include <cub/cub.cuh>
#include <chrono>
#include <unordered_map>

namespace px_remesh {
struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};
#define RM_CHECK(x)                                                                                          \
    do {                                                                                                     \
        cudaError_t e_ = (x);                                                                                \
        if (e_ != cudaSuccess) {                                                                             \
            (void)cudaGetLastError();                                                                        \
            throw Error(std::string(#x ": ") + cudaGetErrorString(e_));                                      \
        }                                                                                                    \
    } while (0)

// Stream-ordered allocations, each borrowed from the engine's budget ledger
// (or capped at `limit` without an engine). Never waits for the engine.
struct Arena {
    cudaStream_t stream{};
    Device *engine = nullptr;
    size_t limit = 0, used = 0, peak = 0, freed = 0;
    std::unordered_map<void *, size_t> live;
    void lend(size_t bytes) {
        if (!engine) {
            if (used + bytes > limit)
                throw Error("remesh needs more than the free GPU memory");
            return;
        }
        std::lock_guard<std::mutex> lock(engine->ledger);
        if (engine->live + engine->lent + bytes > engine->budget)
            throw Error("remesh needs " + std::to_string((used + bytes) >> 20) +
                        " MiB, more than the engine's budget has left");
        engine->lent += bytes;
    }
    void give_back(size_t bytes) {
        if (!engine || !bytes)
            return;
        {
            std::lock_guard<std::mutex> lock(engine->ledger);
            engine->lent -= bytes;
        }
        engine->returned.notify_all();
    }
    template <class T> T *alloc(size_t count) {
        size_t bytes = std::max<size_t>(count * sizeof(T), 16);
        try {
            lend(bytes);
        } catch (const Error &) {
            if (!freed)
                throw;
            sync(); // settle released memory, then ask once more
            lend(bytes);
        }
        void *p = nullptr;
        cudaError_t status = cudaMallocAsync(&p, bytes, stream);
        if (status != cudaSuccess) {
            give_back(bytes);
            RM_CHECK(status);
        }
        live[p] = bytes;
        used += bytes;
        peak = std::max(peak, used);
        return static_cast<T *>(p);
    }
    // Memory returns to the ledger at the next stream sync, once the stream
    // has actually freed it.
    void release(void *p) {
        auto it = live.find(p);
        if (it == live.end())
            return;
        size_t bytes = it->second;
        live.erase(it);
        (void)cudaFreeAsync(p, stream);
        used -= bytes;
        freed += bytes;
    }
    void sync() {
        RM_CHECK(cudaStreamSynchronize(stream));
        give_back(freed);
        freed = 0;
    }
    // Highest priority: these small kernels run beside the flow's large ones
    // and would otherwise wait behind them for free SMs.
    void open_stream() {
        int least = 0, greatest = 0;
        RM_CHECK(cudaDeviceGetStreamPriorityRange(&least, &greatest));
        RM_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, greatest));
    }
    template <class T> T scalar(const T *p) {
        T value{};
        RM_CHECK(cudaMemcpyAsync(&value, p, sizeof(T), cudaMemcpyDeviceToHost, stream));
        sync();
        return value;
    }
    ~Arena() {
        for (auto &item : live)
            (void)cudaFreeAsync(item.first, stream);
        if (stream) {
            (void)cudaStreamSynchronize(stream);
            (void)cudaStreamDestroy(stream);
        }
        give_back(used + freed);
    }
};

struct Box {
    float lo[3], hi[3];
};
// Internal node: its own box; children >= 0 internal, < 0 ~triangle.
struct alignas(16) Node {
    Box box;
    int next[2];
};
struct Tree {
    const Node *nodes;
    const float *triangles; // 9 floats per triangle, in Morton order
};

__device__ __forceinline__ float box_dsq(const Box &a, float px, float py, float pz) {
    // aabb_dist_sq from common/closest_point_bvh.h.
    float dx = (px < a.lo[0]) ? (a.lo[0] - px) : (px > a.hi[0] ? px - a.hi[0] : 0.0f);
    float dy = (py < a.lo[1]) ? (a.lo[1] - py) : (py > a.hi[1] ? py - a.hi[1] : 0.0f);
    float dz = (pz < a.lo[2]) ? (a.lo[2] - pz) : (pz > a.hi[2] ? pz - a.hi[2] : 0.0f);
    return dx * dx + dy * dy + dz * dz;
}
// closest_point_on_triangle from common/closest_point_bvh.h, then |c - p|^2,
// with the same float operation order.
__device__ float triangle_dsq(const float *tri, float px, float py, float pz) {
    float v0x = tri[0], v0y = tri[1], v0z = tri[2];
    float e1x = tri[3] - v0x, e1y = tri[4] - v0y, e1z = tri[5] - v0z;
    float e2x = tri[6] - v0x, e2y = tri[7] - v0y, e2z = tri[8] - v0z;
    float dx = v0x - px, dy = v0y - py, dz = v0z - pz;
    float a = e1x * e1x + e1y * e1y + e1z * e1z;
    float b = e1x * e2x + e1y * e2y + e1z * e2z;
    float c = e2x * e2x + e2y * e2y + e2z * e2z;
    float dd = e1x * dx + e1y * dy + e1z * dz;
    float ee = e2x * dx + e2y * dy + e2z * dz;
    float det = a * c - b * b;
    float s = b * ee - c * dd;
    float t = b * dd - a * ee;
    if (s + t <= det) {
        if (s < 0.0f) {
            if (t < 0.0f) {
                if (dd < 0.0f) {
                    t = 0.0f;
                    s = (-dd >= a) ? 1.0f : -dd / a;
                } else {
                    s = 0.0f;
                    t = (ee >= 0.0f) ? 0.0f : (-ee >= c ? 1.0f : -ee / c);
                }
            } else {
                s = 0.0f;
                t = (ee >= 0.0f) ? 0.0f : (-ee >= c ? 1.0f : -ee / c);
            }
        } else if (t < 0.0f) {
            t = 0.0f;
            s = (dd >= 0.0f) ? 0.0f : (-dd >= a ? 1.0f : -dd / a);
        } else {
            float inv_det = 1.0f / det;
            s *= inv_det;
            t *= inv_det;
        }
    } else {
        if (s < 0.0f) {
            float tmp0 = b + dd;
            float tmp1 = c + ee;
            if (tmp1 > tmp0) {
                float numer = tmp1 - tmp0;
                float denom = a - 2.0f * b + c;
                s = (numer >= denom) ? 1.0f : numer / denom;
                t = 1.0f - s;
            } else {
                s = 0.0f;
                t = (tmp1 <= 0.0f) ? 1.0f : (ee >= 0.0f ? 0.0f : -ee / c);
            }
        } else if (t < 0.0f) {
            float tmp0 = b + ee;
            float tmp1 = a + dd;
            if (tmp1 > tmp0) {
                float numer = tmp1 - tmp0;
                float denom = a - 2.0f * b + c;
                t = (numer >= denom) ? 1.0f : numer / denom;
                s = 1.0f - t;
            } else {
                t = 0.0f;
                s = (tmp1 <= 0.0f) ? 1.0f : (dd >= 0.0f ? 0.0f : -dd / a);
            }
        } else {
            float numer = (c + ee) - (b + dd);
            if (numer <= 0.0f) {
                s = 0.0f;
            } else {
                float denom = a - 2.0f * b + c;
                s = (numer >= denom) ? 1.0f : numer / denom;
            }
            t = 1.0f - s;
        }
    }
    float cx = (v0x + e1x * s) + e2x * t, cy = (v0y + e1y * s) + e2y * t, cz = (v0z + e1z * s) + e2z * t;
    float qx = cx - px, qy = cy - py, qz = cz - pz;
    return qx * qx + qy * qy + qz * qz;
}
// Distance to the closest triangle. Only the minimum is used, so the result
// does not depend on the tree or the visit order. A triangle child is
// measured directly instead of through its box.
__device__ float closest_distance(const Tree &tree, float px, float py, float pz) {
    constexpr int kStack = 96;
    int stack[kStack];
    float bound[kStack];
    int sp = 0;
    float best = __int_as_float(0x7f800000);
    stack[sp] = 0;
    bound[sp++] = 0.f;
    while (sp > 0) {
        --sp;
        if (bound[sp] > best)
            continue;
        const Node &node = tree.nodes[stack[sp]];
        int child[2] = {node.next[0], node.next[1]};
        float d[2];
        for (int k = 0; k < 2; ++k) {
            if (child[k] < 0) {
                float dsq = triangle_dsq(tree.triangles + size_t(~child[k]) * 9, px, py, pz);
                if (dsq < best)
                    best = dsq;
                d[k] = __int_as_float(0x7f800000);
            } else
                d[k] = box_dsq(tree.nodes[child[k]].box, px, py, pz);
        }
        // Nearer child on top of the stack.
        int near = d[1] < d[0], far = 1 - near;
        if (d[far] <= best && sp < kStack) {
            stack[sp] = child[far];
            bound[sp++] = d[far];
        }
        if (d[near] <= best && sp < kStack) {
            stack[sp] = child[near];
            bound[sp++] = d[near];
        }
    }
    return sqrtf(best);
}

__device__ __forceinline__ unsigned expand_bits(unsigned v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}
// (30-bit centroid Morton code << 32) | face: unique keys, ties by face id.
__global__ void morton_keys(unsigned long long *keys, const float *v, const int *f, int nf, float3 lo,
                            float3 inv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nf)
        return;
    const float *a = v + size_t(f[i * 3]) * 3, *b = v + size_t(f[i * 3 + 1]) * 3, *c = v + size_t(f[i * 3 + 2]) * 3;
    float q[3] = {(a[0] + b[0] + c[0]) / 3.f, (a[1] + b[1] + c[1]) / 3.f, (a[2] + b[2] + c[2]) / 3.f};
    float l[3] = {lo.x, lo.y, lo.z}, s[3] = {inv.x, inv.y, inv.z};
    unsigned code = 0;
    for (int d = 0; d < 3; ++d) {
        float u = fminf(fmaxf((q[d] - l[d]) * s[d], 0.f), 1023.f);
        code |= expand_bits(unsigned(u)) << (2 - d);
    }
    keys[i] = (unsigned long long)code << 32 | unsigned(i);
}
__global__ void gather_triangles(float *tris, const unsigned long long *keys, const float *v, const int *f,
                                 int nf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nf)
        return;
    int face = int(keys[i] & 0xffffffffu);
    for (int k = 0; k < 3; ++k)
        for (int d = 0; d < 3; ++d)
            tris[size_t(i) * 9 + k * 3 + d] = v[size_t(f[face * 3 + k]) * 3 + d];
}
__device__ __forceinline__ int key_delta(const unsigned long long *keys, int n, int i, int j) {
    return j < 0 || j >= n ? -1 : __clzll(keys[i] ^ keys[j]);
}
// Karras 2012 hierarchy over unique sorted keys.
__global__ void karras(Node *nodes, int *parent, int *leaf_parent, const unsigned long long *keys, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n - 1)
        return;
    int d = key_delta(keys, n, i, i + 1) - key_delta(keys, n, i, i - 1) >= 0 ? 1 : -1;
    int dmin = key_delta(keys, n, i, i - d);
    int lmax = 2;
    while (key_delta(keys, n, i, i + lmax * d) > dmin)
        lmax *= 2;
    int l = 0;
    for (int t = lmax / 2; t >= 1; t /= 2)
        if (key_delta(keys, n, i, i + (l + t) * d) > dmin)
            l += t;
    int j = i + l * d;
    int dnode = key_delta(keys, n, i, j);
    int s = 0, t = l;
    do {
        t = (t + 1) / 2;
        if (key_delta(keys, n, i, i + (s + t) * d) > dnode)
            s += t;
    } while (t > 1);
    int gamma = i + s * d + min(d, 0);
    int left = min(i, j) == gamma ? ~gamma : gamma, right = max(i, j) == gamma + 1 ? ~(gamma + 1) : gamma + 1;
    nodes[i].next[0] = left;
    nodes[i].next[1] = right;
    if (left < 0)
        leaf_parent[~left] = i;
    else
        parent[left] = i;
    if (right < 0)
        leaf_parent[~right] = i;
    else
        parent[right] = i;
    if (i == 0)
        parent[0] = -1;
}
__device__ __forceinline__ Box triangle_box(const float *t) {
    Box box;
    for (int d = 0; d < 3; ++d) {
        box.lo[d] = fminf(fminf(t[d], t[3 + d]), t[6 + d]);
        box.hi[d] = fmaxf(fmaxf(t[d], t[3 + d]), t[6 + d]);
    }
    return box;
}
// Bottom-up bounds: the second child to finish builds its parent's box from
// its own and the sibling's (a triangle, or a node stored before its thread
// signalled). Min/max are exact, so boxes do not depend on arrival order.
__global__ void refit(Node *nodes, int *visits, const int *parent, const int *leaf_parent, const float *tris,
                      int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    Box box = triangle_box(tris + size_t(i) * 9);
    int child = ~i;
    for (int p = leaf_parent[i]; p >= 0; p = parent[p]) {
        __threadfence();
        if (atomicAdd(visits + p, 1) == 0)
            return;
        __threadfence();
        int sibling = nodes[p].next[0] == child ? nodes[p].next[1] : nodes[p].next[0];
        Box other;
        if (sibling < 0)
            other = triangle_box(tris + size_t(~sibling) * 9);
        else {
            const volatile Box *stored = &nodes[sibling].box;
            for (int d = 0; d < 3; ++d) {
                other.lo[d] = stored->lo[d];
                other.hi[d] = stored->hi[d];
            }
        }
        for (int d = 0; d < 3; ++d) {
            box.lo[d] = fminf(box.lo[d], other.lo[d]);
            box.hi[d] = fmaxf(box.hi[d], other.hi[d]);
        }
        volatile Box *mine = &nodes[p].box;
        for (int d = 0; d < 3; ++d) {
            mine->lo[d] = box.lo[d];
            mine->hi[d] = box.hi[d];
        }
        child = p;
    }
}

// Voxel keys: 15 bits of the 32^3 root cell (x, y, z row-major, the CPU's
// seed order) followed by one xyz octant triple per finer level (the CPU's
// child order). Every candidate list is therefore sorted by key.
__device__ __forceinline__ void decode(unsigned key, int levels, int &x, int &y, int &z) {
    unsigned top = key >> (3 * levels);
    x = int(top >> 10);
    y = int((top >> 5) & 31);
    z = int(top & 31);
    for (int b = levels - 1; b >= 0; --b) {
        unsigned o = (key >> (3 * b)) & 7;
        x = x * 2 + int(o >> 2);
        y = y * 2 + int((o >> 1) & 1);
        z = z * 2 + int(o & 1);
    }
}
__device__ __forceinline__ unsigned encode(int x, int y, int z, int levels) {
    unsigned key = unsigned(((x >> levels) * 1024 + (y >> levels) * 32 + (z >> levels))) << (3 * levels);
    for (int b = levels - 1; b >= 0; --b)
        key |= unsigned(((x >> b) & 1) << 2 | ((y >> b) & 1) << 1 | ((z >> b) & 1)) << (3 * b);
    return key;
}
__global__ void seed(unsigned *keys) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 32768)
        keys[i] = unsigned(i);
}
__global__ void narrow_band(int *active, const unsigned *keys, int n, int levels, int level, float scale, float eps,
                            Tree tree) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int x, y, z;
    decode(keys[i], levels, x, y, z);
    float px = (float(x) + .5f) / level * scale - .5f * scale, py = (float(y) + .5f) / level * scale - .5f * scale,
          pz = (float(z) + .5f) / level * scale - .5f * scale;
    active[i] = fabsf(closest_distance(tree, px, py, pz) - eps) < .87f * scale / level;
}
__global__ void refine(unsigned *out, const unsigned *keys, const int *active, const int *pos, int n, int last) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !active[i])
        return;
    if (last)
        out[pos[i]] = keys[i];
    else
        for (int s = 0; s < 8; ++s)
            out[size_t(pos[i]) * 8 + s] = keys[i] * 8 + unsigned(s);
}

// UDF at the 8 corners of every voxel. Shared corners are evaluated once per
// voxel: more queries than a deduplicated corner set, but no hash table,
// and each value depends only on its corner.
__global__ void corner_udf(float *udf, const unsigned *keys, int n, int levels, int resolution, float scale,
                           float eps, Tree tree) {
    size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= size_t(n) * 8)
        return;
    int c[3], s = int(t & 7);
    decode(keys[t >> 3], levels, c[0], c[1], c[2]);
    c[0] += s >> 2;
    c[1] += (s >> 1) & 1;
    c[2] += s & 1;
    udf[t] = closest_distance(tree, (float(c[0]) / resolution - .5f) * scale, (float(c[1]) / resolution - .5f) * scale,
                              (float(c[2]) / resolution - .5f) * scale) -
             eps;
}
// Dual vertex and edge crossings of one voxel, as remesh.cc.
__global__ void contour(float *dual, signed char *intersected, const float *udf, const unsigned *keys, int n,
                        int levels, int resolution, float scale) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int p[3];
    decode(keys[i], levels, p[0], p[1], p[2]);
    float corner[8];
    for (int s = 0; s < 8; ++s)
        corner[s] = udf[size_t(i) * 8 + s];
    float sum[3] = {};
    int count = 0;
    signed char crossing[3] = {};
    for (int axis = 0; axis < 3; ++axis)
        for (int u = 0; u < 2; ++u)
            for (int v = 0; v < 2; ++v) {
                int a[3] = {p[0], p[1], p[2]};
                int orth = 0;
                for (int d = 0; d < 3; ++d)
                    if (d != axis)
                        a[d] += orth++ == 0 ? u : v;
                int b[3] = {a[0], a[1], a[2]};
                ++b[axis];
                float f0 = corner[(a[0] - p[0]) * 4 + (a[1] - p[1]) * 2 + (a[2] - p[2])],
                      f1 = corner[(b[0] - p[0]) * 4 + (b[1] - p[1]) * 2 + (b[2] - p[2])];
                if ((f0 < 0) != (f1 < 0)) {
                    float t = -f0 / (f1 - f0);
                    for (int d = 0; d < 3; ++d)
                        sum[d] += a[d] + (d == axis ? t : 0);
                    ++count;
                    if (u == 1 && v == 1)
                        crossing[axis] = f0 < 0 ? 1 : -1;
                }
            }
    for (int d = 0; d < 3; ++d) {
        dual[size_t(i) * 3 + d] = ((count ? sum[d] / count : p[d] + .5f) / resolution - .5f) * scale;
        intersected[size_t(i) * 3 + d] = crossing[d];
    }
}
__constant__ int kOffsets[3][4][3] = {{{0, 0, 0}, {0, 0, 1}, {0, 1, 1}, {0, 1, 0}},
                                      {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1}},
                                      {{0, 0, 0}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}}};
__device__ int find_voxel(const unsigned *keys, int n, int x, int y, int z, int resolution, int levels) {
    if (x >= resolution || y >= resolution || z >= resolution)
        return -1;
    unsigned key = encode(x, y, z, levels);
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (keys[mid] < key)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo < n && keys[lo] == key ? lo : -1;
}
__device__ bool quad(int *q, const unsigned *keys, int n, int i, int axis, int resolution, int levels) {
    int p[3];
    decode(keys[i], levels, p[0], p[1], p[2]);
    for (int j = 0; j < 4; ++j) {
        q[j] = find_voxel(keys, n, p[0] + kOffsets[axis][j][0], p[1] + kOffsets[axis][j][1],
                          p[2] + kOffsets[axis][j][2], resolution, levels);
        if (q[j] < 0)
            return false;
    }
    return true;
}
__global__ void count_quads(int *quads, const signed char *intersected, const unsigned *keys, int n, int resolution,
                            int levels) {
    size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= size_t(n) * 3)
        return;
    int q[4];
    quads[t] = intersected[t] && quad(q, keys, n, int(t / 3), int(t % 3), resolution, levels);
}
__device__ float alignment(const float *v, const int *q, const int *s) {
    const float *p0 = v + size_t(q[s[0]]) * 3, *p1 = v + size_t(q[s[1]]) * 3, *p2 = v + size_t(q[s[2]]) * 3,
                *p3 = v + size_t(q[s[3]]) * 3;
    float a[3], b[3], c[3], d[3];
    for (int k = 0; k < 3; ++k) {
        a[k] = p1[k] - p0[k];
        b[k] = p2[k] - p0[k];
        c[k] = p2[k] - p1[k];
        d[k] = p3[k] - p1[k];
    }
    float n0[3] = {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
    float n1[3] = {c[1] * d[2] - c[2] * d[1], c[2] * d[0] - c[0] * d[2], c[0] * d[1] - c[1] * d[0]};
    return fabsf(n0[0] * n1[0] + n0[1] * n1[1] + n0[2] * n1[2]);
}
__global__ void emit_quads(int *faces, int *used, const int *quads, const int *pos, const signed char *intersected,
                           const float *dual, const unsigned *keys, int n, int resolution, int levels) {
    size_t t = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= size_t(n) * 3 || !quads[t])
        return;
    int q[4];
    quad(q, keys, n, int(t / 3), int(t % 3), resolution, levels);
    const int split1p[6] = {0, 2, 1, 0, 3, 2}, split1n[6] = {0, 1, 2, 0, 2, 3};
    const int split2p[6] = {0, 3, 1, 3, 2, 1}, split2n[6] = {0, 1, 3, 3, 1, 2};
    const int *s1 = intersected[t] == 1 ? split1p : split1n, *s2 = intersected[t] == 1 ? split2p : split2n;
    const int *split = alignment(dual, q, s1) > alignment(dual, q, s2) ? s1 : s2;
    for (int j = 0; j < 6; ++j) {
        faces[size_t(pos[t]) * 6 + j] = q[split[j]];
        used[q[split[j]]] = 1;
    }
}
__global__ void compact_vertices(float *out, const float *dual, const int *used, const int *index, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !used[i])
        return;
    for (int d = 0; d < 3; ++d)
        out[size_t(index[i]) * 3 + d] = dual[size_t(i) * 3 + d];
}
__global__ void remap_faces(int *faces, const int *index, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n)
        faces[i] = index[faces[i]];
}

inline unsigned blocks(size_t n) { return unsigned((n + 255) / 256); }
// Exclusive prefix sum of int flags; returns the total.
inline int scan(Arena &arena, const int *in, int *out, int n) {
    size_t bytes = 0;
    RM_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, bytes, in, out, n, arena.stream));
    void *temp = arena.alloc<char>(bytes);
    RM_CHECK(cub::DeviceScan::ExclusiveSum(temp, bytes, in, out, n, arena.stream));
    arena.release(temp);
    return arena.scalar(out + n - 1) + arena.scalar(in + n - 1);
}

inline void run(const px_remesh_request &r, px_remesh_result &result) {
    using clock = std::chrono::steady_clock;
    auto since = [](clock::time_point t) { return std::chrono::duration<double>(clock::now() - t).count(); };
    auto t0 = clock::now();
    int R = r.resolution, nf = r.num_faces, nv = r.num_vertices;
    if (R < 32 || R > 1024 || (R & (R - 1)))
        throw Error("remesh resolution must be a power of two in [32, 1024]");
    if (nf < 2 || nv < 3)
        throw Error("remesh needs at least two triangles");
    int levels = 0;
    while ((32 << levels) < R)
        ++levels;
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

    // LBVH over the input triangles.
    float lo[3] = {INFINITY, INFINITY, INFINITY}, hi[3] = {-INFINITY, -INFINITY, -INFINITY};
    for (int i = 0; i < nv; ++i)
        for (int d = 0; d < 3; ++d) {
            lo[d] = std::min(lo[d], r.vertices[size_t(i) * 3 + d]);
            hi[d] = std::max(hi[d], r.vertices[size_t(i) * 3 + d]);
        }
    float *v = arena.alloc<float>(size_t(nv) * 3);
    int *f = arena.alloc<int>(size_t(nf) * 3);
    RM_CHECK(cudaMemcpyAsync(v, r.vertices, size_t(nv) * 12, cudaMemcpyHostToDevice, s));
    RM_CHECK(cudaMemcpyAsync(f, r.faces, size_t(nf) * 12, cudaMemcpyHostToDevice, s));
    auto *keys = arena.alloc<unsigned long long>(nf), *sorted = arena.alloc<unsigned long long>(nf);
    float3 origin = make_float3(lo[0], lo[1], lo[2]);
    float3 inv = make_float3(1023.f / std::max(hi[0] - lo[0], 1e-20f), 1023.f / std::max(hi[1] - lo[1], 1e-20f),
                             1023.f / std::max(hi[2] - lo[2], 1e-20f));
    morton_keys<<<blocks(nf), 256, 0, s>>>(keys, v, f, nf, origin, inv);
    size_t bytes = 0;
    RM_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, bytes, keys, sorted, nf, 0, 64, s));
    void *temp = arena.alloc<char>(bytes);
    RM_CHECK(cub::DeviceRadixSort::SortKeys(temp, bytes, keys, sorted, nf, 0, 64, s));
    arena.release(temp);
    float *tris = arena.alloc<float>(size_t(nf) * 9);
    gather_triangles<<<blocks(nf), 256, 0, s>>>(tris, sorted, v, f, nf);
    arena.release(v);
    arena.release(f);
    arena.release(keys);
    Node *nodes = arena.alloc<Node>(nf - 1);
    int *parent = arena.alloc<int>(nf - 1), *leaf_parent = arena.alloc<int>(nf), *visits = arena.alloc<int>(nf - 1);
    karras<<<blocks(nf - 1), 256, 0, s>>>(nodes, parent, leaf_parent, sorted, nf);
    RM_CHECK(cudaMemsetAsync(visits, 0, size_t(nf - 1) * 4, s));
    refit<<<blocks(nf), 256, 0, s>>>(nodes, visits, parent, leaf_parent, tris, nf);
    RM_CHECK(cudaGetLastError());
    arena.release(sorted);
    arena.release(parent);
    arena.release(leaf_parent);
    arena.release(visits);
    RM_CHECK(cudaStreamSynchronize(s));
    double t_bvh = since(t0);
    Tree tree{nodes, tris};

    // Narrow-band levels, as remesh.cc.
    auto t1 = clock::now();
    float scale = float(R + 3) / R, eps = scale / R;
    int n = 32768;
    unsigned *cells = arena.alloc<unsigned>(n);
    seed<<<blocks(n), 256, 0, s>>>(cells);
    for (int level = 32, depth = 0;; level *= 2, ++depth) {
        int *active = arena.alloc<int>(n), *pos = arena.alloc<int>(n);
        narrow_band<<<blocks(n), 256, 0, s>>>(active, cells, n, depth, level, scale, eps, tree);
        RM_CHECK(cudaGetLastError());
        int kept = scan(arena, active, pos, n);
        bool last = level == R;
        int next_n = last ? kept : kept * 8;
        unsigned *next = arena.alloc<unsigned>(std::max(next_n, 1));
        refine<<<blocks(n), 256, 0, s>>>(next, cells, active, pos, n, last);
        arena.release(active);
        arena.release(pos);
        arena.release(cells);
        cells = next;
        n = next_n;
        std::fprintf(stderr, "Pixal3D remesh grid %d: %d active candidates\n", level, n);
        if (!n)
            throw Error("Remesh found no surface");
        if (last)
            break;
    }
    double t_levels = since(t1);

    // Corner UDF per voxel.
    t1 = clock::now();
    float *udf = arena.alloc<float>(size_t(n) * 8);
    corner_udf<<<blocks(size_t(n) * 8), 256, 0, s>>>(udf, cells, n, levels, R, scale, eps, tree);
    RM_CHECK(cudaGetLastError());
    RM_CHECK(cudaStreamSynchronize(s));
    arena.release(nodes);
    arena.release(tris);
    double t_udf = since(t1);

    // Dual contouring and compaction, keeping the CPU's voxel/axis order.
    t1 = clock::now();
    float *dual = arena.alloc<float>(size_t(n) * 3);
    auto *intersected = arena.alloc<signed char>(size_t(n) * 3);
    contour<<<blocks(n), 256, 0, s>>>(dual, intersected, udf, cells, n, levels, R, scale);
    RM_CHECK(cudaGetLastError());
    arena.release(udf);
    int *quads = arena.alloc<int>(size_t(n) * 3), *quad_pos = arena.alloc<int>(size_t(n) * 3);
    count_quads<<<blocks(size_t(n) * 3), 256, 0, s>>>(quads, intersected, cells, n, R, levels);
    int nq = scan(arena, quads, quad_pos, n * 3);
    if (!nq)
        throw Error("Remesh produced no faces");
    int *faces = arena.alloc<int>(size_t(nq) * 6), *used = arena.alloc<int>(n), *index = arena.alloc<int>(n);
    RM_CHECK(cudaMemsetAsync(used, 0, size_t(n) * 4, s));
    emit_quads<<<blocks(size_t(n) * 3), 256, 0, s>>>(faces, used, quads, quad_pos, intersected, dual, cells, n, R,
                                                     levels);
    arena.release(quads);
    arena.release(quad_pos);
    arena.release(intersected);
    int kept = scan(arena, used, index, n);
    float *vertices = arena.alloc<float>(size_t(kept) * 3);
    compact_vertices<<<blocks(n), 256, 0, s>>>(vertices, dual, used, index, n);
    remap_faces<<<blocks(size_t(nq) * 6), 256, 0, s>>>(faces, index, size_t(nq) * 6);
    RM_CHECK(cudaGetLastError());
    result.vertices = static_cast<float *>(std::malloc(size_t(kept) * 12));
    result.faces = static_cast<int32_t *>(std::malloc(size_t(nq) * 24));
    if (!result.vertices || !result.faces)
        throw Error("Cannot allocate the remeshed mesh");
    RM_CHECK(cudaMemcpyAsync(result.vertices, vertices, size_t(kept) * 12, cudaMemcpyDeviceToHost, s));
    RM_CHECK(cudaMemcpyAsync(result.faces, faces, size_t(nq) * 24, cudaMemcpyDeviceToHost, s));
    RM_CHECK(cudaStreamSynchronize(s));
    result.num_vertices = kept;
    result.num_faces = nq * 2;
    std::fprintf(stderr,
                 "Pixal3D remesh (GPU): bvh %.2fs, levels %.2fs, udf %.2fs, contour %.2fs, total "
                 "%.2fs, peak %.0f MiB\n",
                 t_bvh, t_levels, t_udf, since(t1), since(t0), arena.peak / 1048576.0);
}
} // namespace px_remesh

extern "C" int px_gpu_remesh(const px_remesh_request *request, px_remesh_result *result) {
    *result = px_remesh_result{};
    try {
        px_remesh::run(*request, *result);
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
