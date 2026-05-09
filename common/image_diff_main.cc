// image_diff: compare two AOV EXRs produced by preview_render and emit
// numeric JSON diff (LLM-readable, no VLM needed).
//
// Usage:
//   image_diff --a a.exr --b b.exr --json diff.json
//              [--bins 32] [--mat mesh.safetensors]
//
// `--mat` reads `face_material_id` (I32 [F]) from the original mesh so the
// diff JSON includes per-region (per-material) deltas. Without it, the
// diff still emits all global metrics; the regions list is empty unless
// the EXR is large enough that face_id alone already segments well.

#define SAFETENSORS_IMPLEMENTATION

#include "image_stats.h"
#include "preview_render.h"
#include "safetensors.h"
#include "tinyexr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

// Find a channel index by name in EXRImage's header arrays.
int findChan(const EXRHeader& h, const char* name) {
  for (int i = 0; i < h.num_channels; ++i) {
    if (std::strcmp(h.channels[i].name, name) == 0) return i;
  }
  return -1;
}

bool loadAOV(const char* path, trellis2::AOVImage& img, bool& has_face_id) {
  EXRVersion ver;
  int rc = ParseEXRVersionFromFile(&ver, path);
  if (rc != TINYEXR_SUCCESS) {
    std::fprintf(stderr, "ParseEXRVersion(%s) failed\n", path);
    return false;
  }
  EXRHeader header; InitEXRHeader(&header);
  const char* err = nullptr;
  rc = ParseEXRHeaderFromFile(&header, &ver, path, &err);
  if (rc != TINYEXR_SUCCESS) {
    std::fprintf(stderr, "ParseEXRHeader(%s): %s\n", path, err ? err : "?");
    FreeEXRErrorMessage(err);
    return false;
  }
  // Force all channels to FLOAT for easy reading.
  for (int i = 0; i < header.num_channels; ++i) {
    if (header.pixel_types[i] == TINYEXR_PIXELTYPE_HALF) {
      header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    }
  }
  EXRImage image; InitEXRImage(&image);
  rc = LoadEXRImageFromFile(&image, &header, path, &err);
  if (rc != TINYEXR_SUCCESS) {
    std::fprintf(stderr, "LoadEXRImage(%s): %s\n", path, err ? err : "?");
    FreeEXRErrorMessage(err);
    FreeEXRHeader(&header);
    return false;
  }
  uint32_t W = (uint32_t)image.width, H = (uint32_t)image.height;
  img.allocate(W, H);
  size_t N = (size_t)W * H;

  int iR = findChan(header, "R"), iG = findChan(header, "G"), iB = findChan(header, "B");
  int iA = findChan(header, "A"), iZ = findChan(header, "Z");
  int iAR = findChan(header, "albedo.R");
  int iAG = findChan(header, "albedo.G");
  int iAB = findChan(header, "albedo.B");
  int iNX = findChan(header, "normal.X");
  int iNY = findChan(header, "normal.Y");
  int iNZ = findChan(header, "normal.Z");
  int iPX = findChan(header, "P.X"), iPY = findChan(header, "P.Y"), iPZ = findChan(header, "P.Z");
  int iFI = findChan(header, "face_id");

  auto chanf = [&](int idx) -> const float* {
    if (idx < 0) return nullptr;
    return reinterpret_cast<const float*>(image.images[idx]);
  };
  const float* cR = chanf(iR), *cG = chanf(iG), *cB = chanf(iB);
  const float* cA = chanf(iA), *cZ = chanf(iZ);
  const float* cAR = chanf(iAR), *cAG = chanf(iAG), *cAB = chanf(iAB);
  const float* cNX = chanf(iNX), *cNY = chanf(iNY), *cNZ = chanf(iNZ);
  const float* cPX = chanf(iPX), *cPY = chanf(iPY), *cPZ = chanf(iPZ);
  const float* cFI = chanf(iFI);

  for (size_t i = 0; i < N; ++i) {
    if (cR) img.beauty[3*i+0] = cR[i];
    if (cG) img.beauty[3*i+1] = cG[i];
    if (cB) img.beauty[3*i+2] = cB[i];
    if (cAR) img.albedo[3*i+0] = cAR[i];
    if (cAG) img.albedo[3*i+1] = cAG[i];
    if (cAB) img.albedo[3*i+2] = cAB[i];
    if (cNX) img.normal[3*i+0] = cNX[i];
    if (cNY) img.normal[3*i+1] = cNY[i];
    if (cNZ) img.normal[3*i+2] = cNZ[i];
    if (cPX) img.position[3*i+0] = cPX[i];
    if (cPY) img.position[3*i+1] = cPY[i];
    if (cPZ) img.position[3*i+2] = cPZ[i];
    if (cZ)  img.depth[i] = cZ[i];
    if (cA)  img.mask[i]  = cA[i];
    if (cFI) img.face_id[i] = (int32_t)cFI[i];
  }
  has_face_id = (cFI != nullptr);
  FreeEXRImage(&image);
  FreeEXRHeader(&header);
  return true;
}

bool loadFaceMaterialId(const char* path, std::vector<int32_t>& out) {
  st_context* ctx = safetensors_open(path);
  if (!ctx) {
    std::fprintf(stderr, "safetensors_open(%s) failed\n", path);
    return false;
  }
  int idx = safetensors_find(ctx, "face_material_id");
  if (idx < 0) { safetensors_close(ctx); return false; }
  if (std::strcmp(safetensors_dtype(ctx, idx), "I32") != 0) {
    safetensors_close(ctx); return false;
  }
  size_t nb = safetensors_nbytes(ctx, idx);
  const int32_t* p = (const int32_t*)safetensors_data(ctx, idx);
  out.assign(p, p + nb / sizeof(int32_t));
  safetensors_close(ctx);
  return true;
}

void usage(const char* a0) {
  std::fprintf(stderr,
    "usage: %s --a a.exr --b b.exr --json diff.json [--bins 32] [--mat mesh.safetensors]\n",
    a0);
}

}  // namespace

int main(int argc, char** argv) {
  const char* a_path = nullptr;
  const char* b_path = nullptr;
  const char* json_path = nullptr;
  const char* mat_path = nullptr;
  int bins = 32;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* opt) -> const char* {
      if (i + 1 >= argc) { std::fprintf(stderr, "missing value for %s\n", opt); std::exit(2); }
      return argv[++i];
    };
    if      (a == "--a")    a_path = need("--a");
    else if (a == "--b")    b_path = need("--b");
    else if (a == "--json") json_path = need("--json");
    else if (a == "--bins") bins = std::atoi(need("--bins"));
    else if (a == "--mat")  mat_path = need("--mat");
    else { usage(argv[0]); return 2; }
  }
  if (!a_path || !b_path || !json_path) { usage(argv[0]); return 2; }

  trellis2::AOVImage A, B;
  bool a_has_fid = false, b_has_fid = false;
  if (!loadAOV(a_path, A, a_has_fid)) return 1;
  if (!loadAOV(b_path, B, b_has_fid)) return 1;
  std::fprintf(stderr, "a: %ux%u (face_id=%s)\n", A.width, A.height, a_has_fid ? "yes" : "no");
  std::fprintf(stderr, "b: %ux%u (face_id=%s)\n", B.width, B.height, b_has_fid ? "yes" : "no");

  std::vector<int32_t> mat;
  if (mat_path) {
    if (!loadFaceMaterialId(mat_path, mat)) {
      std::fprintf(stderr, "warning: --mat %s has no face_material_id; using face_id buckets\n", mat_path);
    } else {
      std::fprintf(stderr, "loaded face_material_id (%zu entries) from %s\n", mat.size(), mat_path);
    }
  }

  trellis2::StatsOpts opts;
  opts.hist_bins = bins;
  trellis2::DiffReport d = trellis2::diff(A, B, mat, opts);
  if (!trellis2::writeDiffJSON(d, json_path)) return 1;
  std::fprintf(stderr, "wrote %s (psnr=%.2fdB mae=%.4g iou=%.3f)\n",
               json_path, d.psnr_db, d.mae, d.mask_iou);
  return 0;
}
