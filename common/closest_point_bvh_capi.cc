// C ABI for ClosestPointBVH so Python (ctypes) can call it.
//
// Build:
//   c++ -std=c++17 -O3 -fPIC -shared -fopenmp \
//       -I. closest_point_bvh_capi.cc lightrt.cc \
//       -o libclosest_point_bvh.so

#include "closest_point_bvh.h"

extern "C" {

void* tr2_cpbvh_create(const float* vertices, uint32_t num_v,
                       const int32_t* faces, uint32_t num_f) {
  auto* bvh = new trellis2::ClosestPointBVH();
  if (!bvh->build(vertices, num_v, faces, num_f)) {
    delete bvh;
    return nullptr;
  }
  return bvh;
}

void tr2_cpbvh_destroy(void* handle) {
  delete reinterpret_cast<trellis2::ClosestPointBVH*>(handle);
}

// Batch unsigned-distance query. Outputs may be NULL if not needed.
void tr2_cpbvh_query(void* handle,
                     const float* points, uint32_t num_p,
                     uint32_t* out_face_id, float* out_dist,
                     float* out_uvw, float* out_closest,
                     uint32_t num_threads) {
  auto* bvh = reinterpret_cast<trellis2::ClosestPointBVH*>(handle);
  bvh->queryBatch(points, num_p, out_face_id, out_dist, out_uvw, out_closest, num_threads);
}

uint32_t tr2_cpbvh_num_triangles(void* handle) {
  auto* bvh = reinterpret_cast<trellis2::ClosestPointBVH*>(handle);
  return bvh->numTriangles();
}

}  // extern "C"
