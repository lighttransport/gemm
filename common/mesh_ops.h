// Mesh editing operations used by trellis2 texgen.
//
// Replaces the trimesh-driven Python steps in `cumesh_xatlas.py`:
//   - compute_vertex_normals
//   - remove_duplicate_faces
//   - remove_small_connected_components
//   - unify_face_orientations  (consistent winding via BFS + outward flip)
//   - fill_holes               (small loops via fan triangulation)
//   - uv_unwrap                (delegates to vendored xatlas.cc)
//
// Design: Mesh held as plain SoA `std::vector` arrays; ops mutate in place.
// All operations are deterministic, OpenMP-friendly where useful.
//
// Pure C++17. Only system <cmath>/<vector>/<unordered_map>; no graphics libs.
// Vendored xatlas (common/xatlas.{h,cc}) is the only third-party dependency
// and is built into the same shared object.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace trellis2 {

struct Mesh {
  // Vertices: [num_v, 3] row-major float32.
  std::vector<float> v;
  // Faces:    [num_f, 3] row-major int32.
  std::vector<int32_t> f;

  uint32_t numV() const { return static_cast<uint32_t>(v.size() / 3); }
  uint32_t numF() const { return static_cast<uint32_t>(f.size() / 3); }

  void set(const float* verts, uint32_t nv, const int32_t* faces, uint32_t nf) {
    v.assign(verts, verts + 3 * nv);
    f.assign(faces, faces + 3 * nf);
  }
};

// ---- vec3 helpers ---------------------------------------------------------

inline void v3sub(const float a[3], const float b[3], float r[3]) {
  r[0] = a[0] - b[0]; r[1] = a[1] - b[1]; r[2] = a[2] - b[2];
}
inline void v3cross(const float a[3], const float b[3], float r[3]) {
  r[0] = a[1] * b[2] - a[2] * b[1];
  r[1] = a[2] * b[0] - a[0] * b[2];
  r[2] = a[0] * b[1] - a[1] * b[0];
}
inline float v3dot(const float a[3], const float b[3]) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// ---- vertex normals (area-weighted, then normalized) ----------------------

inline void computeVertexNormals(const Mesh& m, std::vector<float>& out_n) {
  uint32_t nv = m.numV();
  uint32_t nf = m.numF();
  out_n.assign(3 * nv, 0.0f);
  for (uint32_t fi = 0; fi < nf; ++fi) {
    int32_t i0 = m.f[3 * fi + 0], i1 = m.f[3 * fi + 1], i2 = m.f[3 * fi + 2];
    const float* p0 = &m.v[3 * i0];
    const float* p1 = &m.v[3 * i1];
    const float* p2 = &m.v[3 * i2];
    float e1[3], e2[3], n[3];
    v3sub(p1, p0, e1);
    v3sub(p2, p0, e2);
    v3cross(e1, e2, n);
    for (int k = 0; k < 3; ++k) {
      out_n[3 * i0 + k] += n[k];
      out_n[3 * i1 + k] += n[k];
      out_n[3 * i2 + k] += n[k];
    }
  }
  for (uint32_t i = 0; i < nv; ++i) {
    float* n = &out_n[3 * i];
    float l = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
    if (l > 0.0f) { n[0] /= l; n[1] /= l; n[2] /= l; }
  }
}

// ---- remove duplicate faces (sort triples, drop dups) ---------------------

inline void removeDuplicateFaces(Mesh& m) {
  uint32_t nf = m.numF();
  struct Key { int32_t a, b, c; };
  auto sorted_triple = [&](uint32_t fi) -> Key {
    int32_t a = m.f[3 * fi + 0], b = m.f[3 * fi + 1], c = m.f[3 * fi + 2];
    if (a > b) std::swap(a, b);
    if (b > c) std::swap(b, c);
    if (a > b) std::swap(a, b);
    return {a, b, c};
  };
  // hash on Key
  struct KeyHash {
    size_t operator()(const Key& k) const {
      uint64_t h = (uint64_t)(uint32_t)k.a;
      h = h * 1315423911u + (uint32_t)k.b;
      h = h * 1315423911u + (uint32_t)k.c;
      return std::hash<uint64_t>{}(h);
    }
  };
  struct KeyEq { bool operator()(const Key& a, const Key& b) const {
    return a.a == b.a && a.b == b.b && a.c == b.c; } };
  std::unordered_set<Key, KeyHash, KeyEq> seen;
  seen.reserve(nf);
  std::vector<int32_t> nf_out;
  nf_out.reserve(m.f.size());
  for (uint32_t fi = 0; fi < nf; ++fi) {
    Key k = sorted_triple(fi);
    if (k.a == k.b || k.b == k.c) continue;  // degenerate
    if (seen.insert(k).second) {
      nf_out.push_back(m.f[3 * fi + 0]);
      nf_out.push_back(m.f[3 * fi + 1]);
      nf_out.push_back(m.f[3 * fi + 2]);
    }
  }
  m.f.swap(nf_out);
}

// ---- connected-components on face graph (faces share edge -> same comp) ---

namespace detail {
inline std::vector<uint32_t> faceComponents(const Mesh& m) {
  uint32_t nf = m.numF();
  // Build edge -> face list. Edge keyed by sorted vertex pair.
  using Edge = uint64_t;
  auto enc = [](int32_t a, int32_t b) -> Edge {
    if (a > b) std::swap(a, b);
    return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
  };
  std::unordered_map<Edge, std::vector<uint32_t>> em;
  em.reserve(nf * 2);
  for (uint32_t fi = 0; fi < nf; ++fi) {
    int32_t i0 = m.f[3 * fi + 0], i1 = m.f[3 * fi + 1], i2 = m.f[3 * fi + 2];
    em[enc(i0, i1)].push_back(fi);
    em[enc(i1, i2)].push_back(fi);
    em[enc(i2, i0)].push_back(fi);
  }
  std::vector<uint32_t> comp(nf, UINT32_MAX);
  uint32_t cid = 0;
  std::vector<uint32_t> stack;
  for (uint32_t s = 0; s < nf; ++s) {
    if (comp[s] != UINT32_MAX) continue;
    stack.clear();
    stack.push_back(s);
    comp[s] = cid;
    while (!stack.empty()) {
      uint32_t fi = stack.back(); stack.pop_back();
      int32_t i0 = m.f[3 * fi + 0], i1 = m.f[3 * fi + 1], i2 = m.f[3 * fi + 2];
      Edge es[3] = { enc(i0, i1), enc(i1, i2), enc(i2, i0) };
      for (Edge e : es) {
        for (uint32_t nb : em[e]) {
          if (comp[nb] == UINT32_MAX) {
            comp[nb] = cid;
            stack.push_back(nb);
          }
        }
      }
    }
    ++cid;
  }
  return comp;
}

inline float triArea(const float p0[3], const float p1[3], const float p2[3]) {
  float e1[3], e2[3], n[3];
  v3sub(p1, p0, e1);
  v3sub(p2, p0, e2);
  v3cross(e1, e2, n);
  return 0.5f * std::sqrt(n[0]*n[0] + n[1]*n[1] + n[2]*n[2]);
}
}  // namespace detail

inline void removeSmallConnectedComponents(Mesh& m, float area_threshold) {
  uint32_t nf = m.numF();
  if (nf == 0) return;
  auto comp = detail::faceComponents(m);
  uint32_t nc = comp.empty() ? 0 : (*std::max_element(comp.begin(), comp.end()) + 1);
  std::vector<float> area(nc, 0.0f);
  for (uint32_t fi = 0; fi < nf; ++fi) {
    const float* p0 = &m.v[3 * m.f[3 * fi + 0]];
    const float* p1 = &m.v[3 * m.f[3 * fi + 1]];
    const float* p2 = &m.v[3 * m.f[3 * fi + 2]];
    area[comp[fi]] += detail::triArea(p0, p1, p2);
  }
  // Always keep at least the largest component.
  uint32_t big = (uint32_t)(std::max_element(area.begin(), area.end()) - area.begin());
  std::vector<int32_t> nf_out;
  nf_out.reserve(m.f.size());
  for (uint32_t fi = 0; fi < nf; ++fi) {
    if (area[comp[fi]] >= area_threshold || comp[fi] == big) {
      nf_out.push_back(m.f[3 * fi + 0]);
      nf_out.push_back(m.f[3 * fi + 1]);
      nf_out.push_back(m.f[3 * fi + 2]);
    }
  }
  m.f.swap(nf_out);
}

// ---- consistent face winding via BFS, then global flip if avg normal
//      points inward (relative to centroid) ---------------------------------

inline void unifyFaceOrientations(Mesh& m) {
  uint32_t nf = m.numF();
  if (nf == 0) return;

  // Build directed-edge -> face map; if a neighbor face shares the same
  // *directed* edge, its winding is opposite ours, so flip it.
  using Edge = uint64_t;
  auto enc = [](int32_t a, int32_t b) -> Edge {
    return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
  };
  std::unordered_map<Edge, std::vector<uint32_t>> dir_edges;
  dir_edges.reserve(nf * 3);
  for (uint32_t fi = 0; fi < nf; ++fi) {
    int32_t i0 = m.f[3 * fi + 0], i1 = m.f[3 * fi + 1], i2 = m.f[3 * fi + 2];
    dir_edges[enc(i0, i1)].push_back(fi);
    dir_edges[enc(i1, i2)].push_back(fi);
    dir_edges[enc(i2, i0)].push_back(fi);
  }

  std::vector<uint8_t> visited(nf, 0);
  std::vector<uint32_t> stack;
  for (uint32_t seed = 0; seed < nf; ++seed) {
    if (visited[seed]) continue;
    visited[seed] = 1;
    stack.clear();
    stack.push_back(seed);
    while (!stack.empty()) {
      uint32_t fi = stack.back(); stack.pop_back();
      int32_t i0 = m.f[3 * fi + 0], i1 = m.f[3 * fi + 1], i2 = m.f[3 * fi + 2];
      // For each directed edge of fi, find neighbor sharing same direction
      // (= bad winding) or reverse (= good winding).
      Edge dirs[3] = { enc(i0, i1), enc(i1, i2), enc(i2, i0) };
      Edge revs[3] = { enc(i1, i0), enc(i2, i1), enc(i0, i2) };
      for (int k = 0; k < 3; ++k) {
        for (uint32_t nb : dir_edges[dirs[k]]) {
          if (nb != fi && !visited[nb]) {
            // same direction -> flip nb
            std::swap(m.f[3 * nb + 1], m.f[3 * nb + 2]);
            // Rebuild nb's entries in dir_edges? Simpler: mark visited and
            // descend; subsequent neighbours will use the flipped winding
            // because we already swapped. However, dir_edges still holds the
            // old encoding. To avoid stale lookup, re-emit:
            int32_t a = m.f[3 * nb + 0], b = m.f[3 * nb + 1], c = m.f[3 * nb + 2];
            dir_edges[enc(a, b)].push_back(nb);
            dir_edges[enc(b, c)].push_back(nb);
            dir_edges[enc(c, a)].push_back(nb);
            visited[nb] = 1;
            stack.push_back(nb);
          }
        }
        for (uint32_t nb : dir_edges[revs[k]]) {
          if (nb != fi && !visited[nb]) {
            visited[nb] = 1;
            stack.push_back(nb);
          }
        }
      }
    }
  }

  // Outward-normal heuristic: signed volume via divergence theorem; flip all
  // if negative.
  double vol6 = 0.0;
  for (uint32_t fi = 0; fi < nf; ++fi) {
    const float* p0 = &m.v[3 * m.f[3 * fi + 0]];
    const float* p1 = &m.v[3 * m.f[3 * fi + 1]];
    const float* p2 = &m.v[3 * m.f[3 * fi + 2]];
    vol6 += (double)p0[0] * (p1[1] * p2[2] - p1[2] * p2[1])
          - (double)p0[1] * (p1[0] * p2[2] - p1[2] * p2[0])
          + (double)p0[2] * (p1[0] * p2[1] - p1[1] * p2[0]);
  }
  if (vol6 < 0.0) {
    for (uint32_t fi = 0; fi < nf; ++fi) {
      std::swap(m.f[3 * fi + 1], m.f[3 * fi + 2]);
    }
  }
}

// ---- fill holes by fan triangulation around boundary loops ----------------

inline void fillHoles(Mesh& m, uint32_t max_hole_edges = 64) {
  uint32_t nf = m.numF();
  using Edge = uint64_t;
  auto enc = [](int32_t a, int32_t b) -> Edge {
    return ((uint64_t)(uint32_t)a << 32) | (uint32_t)b;
  };
  std::unordered_map<Edge, int32_t> count;  // count of directed edges (a->b)
  count.reserve(nf * 3);
  for (uint32_t fi = 0; fi < nf; ++fi) {
    int32_t i0 = m.f[3 * fi + 0], i1 = m.f[3 * fi + 1], i2 = m.f[3 * fi + 2];
    count[enc(i0, i1)]++;
    count[enc(i1, i2)]++;
    count[enc(i2, i0)]++;
  }
  // Boundary directed edges: those whose reverse has count 0.
  std::unordered_map<int32_t, int32_t> next_of;  // a -> b for boundary edges
  next_of.reserve(count.size() / 8 + 1);
  for (auto& kv : count) {
    Edge e = kv.first;
    int32_t a = (int32_t)(e >> 32), b = (int32_t)(e & 0xffffffffu);
    if (count.find(enc(b, a)) == count.end()) {
      next_of[a] = b;
    }
  }
  // Walk loops.
  std::unordered_set<int32_t> used;
  for (auto& kv : next_of) {
    int32_t s = kv.first;
    if (used.count(s)) continue;
    std::vector<int32_t> loop;
    int32_t cur = s;
    while (true) {
      auto it = next_of.find(cur);
      if (it == next_of.end()) { loop.clear(); break; }
      if (used.count(cur)) { loop.clear(); break; }
      used.insert(cur);
      loop.push_back(cur);
      cur = it->second;
      if (cur == s) break;
      if (loop.size() > max_hole_edges) { loop.clear(); break; }
    }
    if (loop.size() < 3) continue;
    // Fan triangulate around loop[0].
    int32_t a = loop[0];
    for (size_t k = 1; k + 1 < loop.size(); ++k) {
      m.f.push_back(a);
      m.f.push_back(loop[k]);
      m.f.push_back(loop[k + 1]);
    }
  }
}

// ---- UV unwrap via xatlas (vendored, common/xatlas.{h,cc}) ----------------
//
// Returns: unwrapped vertices (V', 3), faces (F', 3), uvs in [0,1] (V', 2),
// and vmap (V') mapping unwrapped vertex -> original vertex.

bool uvUnwrap(const Mesh& in,
              std::vector<float>& out_v,
              std::vector<int32_t>& out_f,
              std::vector<float>& out_uv,
              std::vector<int32_t>& out_vmap);

}  // namespace trellis2
