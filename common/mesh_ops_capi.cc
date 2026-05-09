// C ABI for trellis2::Mesh ops + UV unwrap, called from Python (ctypes).
//
// Build:
//   c++ -std=c++17 -O3 -fPIC -shared -fopenmp \
//       -I. mesh_ops.cc mesh_ops_capi.cc xatlas.cc \
//       -o libmesh_ops.so

#include "mesh_ops.h"

#include <cstdint>
#include <cstring>

extern "C" {

// ---- Mesh handle ----------------------------------------------------------

void* tr2_mesh_create() {
  return new trellis2::Mesh();
}

void tr2_mesh_destroy(void* h) {
  delete reinterpret_cast<trellis2::Mesh*>(h);
}

void tr2_mesh_set(void* h,
                  const float* verts, uint32_t nv,
                  const int32_t* faces, uint32_t nf) {
  auto* m = reinterpret_cast<trellis2::Mesh*>(h);
  m->set(verts, nv, faces, nf);
}

uint32_t tr2_mesh_num_v(void* h) {
  return reinterpret_cast<trellis2::Mesh*>(h)->numV();
}
uint32_t tr2_mesh_num_f(void* h) {
  return reinterpret_cast<trellis2::Mesh*>(h)->numF();
}

void tr2_mesh_get_v(void* h, float* out) {
  auto* m = reinterpret_cast<trellis2::Mesh*>(h);
  std::memcpy(out, m->v.data(), m->v.size() * sizeof(float));
}
void tr2_mesh_get_f(void* h, int32_t* out) {
  auto* m = reinterpret_cast<trellis2::Mesh*>(h);
  std::memcpy(out, m->f.data(), m->f.size() * sizeof(int32_t));
}

// ---- ops ------------------------------------------------------------------

void tr2_mesh_compute_vertex_normals(void* h, float* out_n) {
  auto* m = reinterpret_cast<trellis2::Mesh*>(h);
  std::vector<float> n;
  trellis2::computeVertexNormals(*m, n);
  std::memcpy(out_n, n.data(), n.size() * sizeof(float));
}

void tr2_mesh_remove_duplicate_faces(void* h) {
  trellis2::removeDuplicateFaces(*reinterpret_cast<trellis2::Mesh*>(h));
}

void tr2_mesh_remove_small_components(void* h, float area_threshold) {
  trellis2::removeSmallConnectedComponents(
      *reinterpret_cast<trellis2::Mesh*>(h), area_threshold);
}

void tr2_mesh_unify_orientations(void* h) {
  trellis2::unifyFaceOrientations(*reinterpret_cast<trellis2::Mesh*>(h));
}

void tr2_mesh_fill_holes(void* h, uint32_t max_hole_edges) {
  trellis2::fillHoles(*reinterpret_cast<trellis2::Mesh*>(h),
                      max_hole_edges == 0 ? 64u : max_hole_edges);
}

// ---- UV unwrap result -----------------------------------------------------

struct Tr2UnwrapResult {
  std::vector<float>   v;
  std::vector<int32_t> f;
  std::vector<float>   uv;
  std::vector<int32_t> vmap;
};

void* tr2_unwrap_create() {
  return new Tr2UnwrapResult();
}

void tr2_unwrap_destroy(void* h) {
  delete reinterpret_cast<Tr2UnwrapResult*>(h);
}

// returns 0 on success, nonzero on failure
int tr2_mesh_uv_unwrap(void* mesh_h, void* unwrap_h) {
  auto* m = reinterpret_cast<trellis2::Mesh*>(mesh_h);
  auto* u = reinterpret_cast<Tr2UnwrapResult*>(unwrap_h);
  bool ok = trellis2::uvUnwrap(*m, u->v, u->f, u->uv, u->vmap);
  return ok ? 0 : -1;
}

uint32_t tr2_unwrap_num_v(void* h) {
  return (uint32_t)(reinterpret_cast<Tr2UnwrapResult*>(h)->v.size() / 3);
}
uint32_t tr2_unwrap_num_f(void* h) {
  return (uint32_t)(reinterpret_cast<Tr2UnwrapResult*>(h)->f.size() / 3);
}

void tr2_unwrap_get_v(void* h, float* out) {
  auto* u = reinterpret_cast<Tr2UnwrapResult*>(h);
  std::memcpy(out, u->v.data(), u->v.size() * sizeof(float));
}
void tr2_unwrap_get_f(void* h, int32_t* out) {
  auto* u = reinterpret_cast<Tr2UnwrapResult*>(h);
  std::memcpy(out, u->f.data(), u->f.size() * sizeof(int32_t));
}
void tr2_unwrap_get_uv(void* h, float* out) {
  auto* u = reinterpret_cast<Tr2UnwrapResult*>(h);
  std::memcpy(out, u->uv.data(), u->uv.size() * sizeof(float));
}
void tr2_unwrap_get_vmap(void* h, int32_t* out) {
  auto* u = reinterpret_cast<Tr2UnwrapResult*>(h);
  std::memcpy(out, u->vmap.data(), u->vmap.size() * sizeof(int32_t));
}

}  // extern "C"
