// Closest-point-on-triangle BVH wrapper around lightrt SBVH.
//
// Used by trellis2 texgen to map texel positions on the simplified mesh back
// to the closest point on the original (pre-simplify) mesh. Drop-in for
// `cumesh.cuBVH.unsigned_distance(points, return_uvw=True)`.
//
// Header-only; #include after "lightrt.hh".

#pragma once

#include "lightrt.hh"

#include <cmath>
#include <cstdint>
#include <vector>

namespace trellis2 {

struct ClosestPointHit {
  uint32_t face_id;   // Index into the original triangle list (-1 if no hit).
  float    distance;  // Euclidean distance to the closest point.
  float    closest[3];
  float    uvw[3];    // Barycentric: closest = uvw[0]*v0 + uvw[1]*v1 + uvw[2]*v2.
};

// Squared distance from point p to AABB.
static inline float aabb_dist_sq(const lightrt::AABB& a, const lightrt::Vec3& p) noexcept {
  float dx = (p.x < a.min.x) ? (a.min.x - p.x) : (p.x > a.max.x ? p.x - a.max.x : 0.0f);
  float dy = (p.y < a.min.y) ? (a.min.y - p.y) : (p.y > a.max.y ? p.y - a.max.y : 0.0f);
  float dz = (p.z < a.min.z) ? (a.min.z - p.z) : (p.z > a.max.z ? p.z - a.max.z : 0.0f);
  return dx * dx + dy * dy + dz * dz;
}

// Closest point from p to triangle (v0,v1,v2). Sets w0/w1/w2 such that
// closest = w0*v0 + w1*v1 + w2*v2 and w0+w1+w2 = 1.
// Reference: Eberly, "Distance Between Point and Triangle in 3D".
static inline void closest_point_on_triangle(
    const lightrt::Vec3& p,
    const lightrt::Vec3& v0,
    const lightrt::Vec3& v1,
    const lightrt::Vec3& v2,
    lightrt::Vec3& closest,
    float& w0, float& w1, float& w2) noexcept {
  lightrt::Vec3 e1 = v1 - v0;
  lightrt::Vec3 e2 = v2 - v0;
  lightrt::Vec3 d  = v0 - p;
  float a = e1.dot(e1);
  float b = e1.dot(e2);
  float c = e2.dot(e2);
  float dd = e1.dot(d);
  float ee = e2.dot(d);
  float det = a * c - b * b;
  float s = b * ee - c * dd;
  float t = b * dd - a * ee;

  if (s + t <= det) {
    if (s < 0.0f) {
      if (t < 0.0f) {
        // region 4
        if (dd < 0.0f) {
          t = 0.0f;
          s = (-dd >= a) ? 1.0f : -dd / a;
        } else {
          s = 0.0f;
          t = (ee >= 0.0f) ? 0.0f : (-ee >= c ? 1.0f : -ee / c);
        }
      } else {
        // region 3
        s = 0.0f;
        t = (ee >= 0.0f) ? 0.0f : (-ee >= c ? 1.0f : -ee / c);
      }
    } else if (t < 0.0f) {
      // region 5
      t = 0.0f;
      s = (dd >= 0.0f) ? 0.0f : (-dd >= a ? 1.0f : -dd / a);
    } else {
      // region 0 (interior)
      float inv_det = 1.0f / det;
      s *= inv_det;
      t *= inv_det;
    }
  } else {
    if (s < 0.0f) {
      // region 2
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
      // region 6
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
      // region 1
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

  closest = v0 + e1 * s + e2 * t;
  w1 = s;
  w2 = t;
  w0 = 1.0f - s - t;
}

class ClosestPointBVH {
public:
  ClosestPointBVH() = default;

  // Build SBVH from raw vertices/faces.
  // vertices: [num_v, 3] float32 row-major
  // faces:    [num_f, 3] int32 row-major
  bool build(const float* vertices, uint32_t num_v,
             const int32_t* faces, uint32_t num_f) {
    triangles_.clear();
    triangles_.reserve(num_f);
    for (uint32_t f = 0; f < num_f; ++f) {
      uint32_t i0 = static_cast<uint32_t>(faces[3 * f + 0]);
      uint32_t i1 = static_cast<uint32_t>(faces[3 * f + 1]);
      uint32_t i2 = static_cast<uint32_t>(faces[3 * f + 2]);
      const float* p0 = vertices + 3 * i0;
      const float* p1 = vertices + 3 * i1;
      const float* p2 = vertices + 3 * i2;
      triangles_.emplace_back(
          lightrt::Vec3(p0[0], p0[1], p0[2]),
          lightrt::Vec3(p1[0], p1[1], p1[2]),
          lightrt::Vec3(p2[0], p2[1], p2[2]));
    }
    lightrt::SBVHBuildConfig cfg;
    return sbvh_.build(triangles_, cfg);
  }

  // Single-point closest-point query.
  ClosestPointHit query(const lightrt::Vec3& p) const noexcept {
    ClosestPointHit best{};
    best.face_id = UINT32_MAX;
    best.distance = std::numeric_limits<float>::infinity();
    float best_dist_sq = std::numeric_limits<float>::infinity();

    const auto& nodes = sbvh_.getNodes();
    const auto& refs  = sbvh_.getReferences();
    if (nodes.empty()) return best;

    // Iterative DFS with a small fixed stack.
    constexpr uint32_t kStackSize = 256;
    uint32_t stack[kStackSize];
    int32_t  sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
      uint32_t idx = stack[--sp];
      const lightrt::BVHNode& node = nodes[idx];
      // Node-AABB lower bound.
      if (aabb_dist_sq(node.bounds, p) >= best_dist_sq) continue;

      if (node.isLeaf()) {
        for (uint32_t i = 0; i < node.prim_count; ++i) {
          const auto& ref = refs[node.prim_offset + i];
          uint32_t pid = ref.prim_id;
          const lightrt::Triangle& tri = triangles_[pid];
          lightrt::Vec3 c;
          float w0, w1, w2;
          closest_point_on_triangle(p, tri.v0, tri.v1, tri.v2, c, w0, w1, w2);
          lightrt::Vec3 diff = c - p;
          float dsq = diff.dot(diff);
          if (dsq < best_dist_sq) {
            best_dist_sq = dsq;
            best.face_id = pid;
            best.closest[0] = c.x; best.closest[1] = c.y; best.closest[2] = c.z;
            best.uvw[0] = w0; best.uvw[1] = w1; best.uvw[2] = w2;
          }
        }
      } else {
        // Visit closer child first for better pruning.
        uint32_t l = node.left_child;
        uint32_t r = node.right_child;
        float dl = aabb_dist_sq(nodes[l].bounds, p);
        float dr = aabb_dist_sq(nodes[r].bounds, p);
        if (dl < dr) {
          if (dr < best_dist_sq && sp < (int32_t)kStackSize) stack[sp++] = r;
          if (dl < best_dist_sq && sp < (int32_t)kStackSize) stack[sp++] = l;
        } else {
          if (dl < best_dist_sq && sp < (int32_t)kStackSize) stack[sp++] = l;
          if (dr < best_dist_sq && sp < (int32_t)kStackSize) stack[sp++] = r;
        }
      }
    }

    best.distance = std::sqrt(best_dist_sq);
    return best;
  }

  // Batch query. points/out_* are flat row-major float arrays (out_face_id is
  // uint32_t, output count == num_p; uvw is 3*num_p; closest is 3*num_p).
  void queryBatch(const float* points, uint32_t num_p,
                  uint32_t* out_face_id, float* out_dist,
                  float* out_uvw, float* out_closest,
                  uint32_t num_threads = 0) const {
#if defined(_OPENMP)
    (void)num_threads;
    #pragma omp parallel for schedule(dynamic, 256)
#else
    (void)num_threads;
#endif
    for (int64_t i = 0; i < (int64_t)num_p; ++i) {
      lightrt::Vec3 p(points[3 * i + 0], points[3 * i + 1], points[3 * i + 2]);
      ClosestPointHit h = query(p);
      out_face_id[i] = h.face_id;
      if (out_dist)    out_dist[i] = h.distance;
      if (out_uvw) {
        out_uvw[3 * i + 0] = h.uvw[0];
        out_uvw[3 * i + 1] = h.uvw[1];
        out_uvw[3 * i + 2] = h.uvw[2];
      }
      if (out_closest) {
        out_closest[3 * i + 0] = h.closest[0];
        out_closest[3 * i + 1] = h.closest[1];
        out_closest[3 * i + 2] = h.closest[2];
      }
    }
  }

  uint32_t numTriangles() const noexcept { return static_cast<uint32_t>(triangles_.size()); }

private:
  lightrt::SBVH sbvh_;
  std::vector<lightrt::Triangle> triangles_;
};

}  // namespace trellis2
