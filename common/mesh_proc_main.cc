// mesh_proc - clean a mesh + (optionally) UV-unwrap, all via safetensors I/O.
//
// Usage:
//   mesh_proc --in input.safetensors --out output.safetensors \
//             [--no-fill-holes] [--no-dedup] [--no-unify] \
//             [--small-area 0.0] [--unwrap]
//
// Pipeline (matches the order in cumesh_xatlas.py / o_voxel.postprocess):
//   1. remove_duplicate_faces
//   2. remove_small_connected_components(area_threshold)  (if >0)
//   3. unify_face_orientations
//   4. fill_holes
//   5. compute_vertex_normals  (always written)
//   6. uv_unwrap               (only with --unwrap; output supersedes 1-4 mesh)

#define SAFETENSORS_IMPLEMENTATION
#define SAFETENSORS_WRITER_IMPLEMENTATION

#include "mesh_io.h"
#include "mesh_ops.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static void usage(const char* argv0) {
  fprintf(stderr,
    "usage: %s --in PATH --out PATH [--no-dedup] [--no-unify] [--no-fill-holes]\n"
    "                [--small-area F] [--unwrap]\n", argv0);
}

int main(int argc, char** argv) {
  const char* in_path = nullptr;
  const char* out_path = nullptr;
  bool do_dedup = true, do_unify = true, do_fill = true, do_unwrap = false;
  float small_area = 0.0f;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* opt) -> const char* {
      if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", opt); std::exit(2); }
      return argv[++i];
    };
    if (a == "--in")              in_path = need("--in");
    else if (a == "--out")        out_path = need("--out");
    else if (a == "--no-dedup")   do_dedup = false;
    else if (a == "--no-unify")   do_unify = false;
    else if (a == "--no-fill-holes") do_fill = false;
    else if (a == "--small-area") small_area = (float)std::atof(need("--small-area"));
    else if (a == "--unwrap")     do_unwrap = true;
    else { usage(argv[0]); return 2; }
  }
  if (!in_path || !out_path) { usage(argv[0]); return 2; }

  trellis2::Mesh m;
  if (!trellis2::loadMesh(in_path, m)) {
    fprintf(stderr, "failed to load %s\n", in_path);
    return 1;
  }
  fprintf(stderr, "in:  V=%u F=%u\n", m.numV(), m.numF());

  if (do_dedup)  trellis2::removeDuplicateFaces(m);
  if (small_area > 0.0f) trellis2::removeSmallConnectedComponents(m, small_area);
  if (do_unify)  trellis2::unifyFaceOrientations(m);
  if (do_fill)   trellis2::fillHoles(m);

  std::vector<float> normals;
  trellis2::computeVertexNormals(m, normals);

  trellis2::MeshWriteOpts opts;
  opts.normals = normals.data();

  std::vector<float>   uv_v, uv_uv;
  std::vector<int32_t> uv_f, uv_vmap;
  trellis2::Mesh out_m;
  if (do_unwrap) {
    if (!trellis2::uvUnwrap(m, uv_v, uv_f, uv_uv, uv_vmap)) {
      fprintf(stderr, "uvUnwrap failed\n");
      return 1;
    }
    out_m.v = std::move(uv_v);
    out_m.f = std::move(uv_f);
    // Recompute normals on the unwrapped (split) mesh.
    trellis2::computeVertexNormals(out_m, normals);
    opts.normals = normals.data();
    opts.uvs     = uv_uv.data();
    opts.vmap    = uv_vmap.data();
    fprintf(stderr, "unwrap: V=%u F=%u\n", out_m.numV(), out_m.numF());
  } else {
    out_m = std::move(m);
  }

  if (!trellis2::saveMesh(out_path, out_m, opts)) {
    fprintf(stderr, "failed to save %s\n", out_path);
    return 1;
  }
  fprintf(stderr, "out: V=%u F=%u -> %s\n", out_m.numV(), out_m.numF(), out_path);
  return 0;
}
