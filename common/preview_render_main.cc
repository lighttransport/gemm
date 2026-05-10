// preview_render: render a textured mesh.safetensors with analytic sun+sky.
//
// Usage:
//   preview_render --in mesh.safetensors [--texture base.png]
//                  [--out beauty.png] [--exr aov.exr]
//                  [-w 800] [-h 600] [--yaw 35] [--pitch 20]
//                  [--sun-yaw 45] [--sun-elev 50] [--turbidity 2.5]
//                  [--exposure 0.0]
//
// Inputs:
//   mesh.safetensors  - schema from common/mesh_io.h (vertices, faces; uvs/vmap optional)
//   base.png          - optional sRGB baseColor texture (decoded to linear).
//
// Outputs:
//   beauty.png        - tonemapped sRGB image
//   aov.exr           - multilayer EXR with beauty/albedo/normal/position/depth/mask

#define SAFETENSORS_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION

#include "image_stats.h"
#include "mesh_io.h"
#include "preview_render.h"
#include "stb_image.h"
#include "stb_image_write.h"
#include "tinyexr.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void usage(const char* argv0) {
  fprintf(stderr,
    "usage: %s --in mesh.safetensors [--texture base.png] [--out beauty.png] [--exr aov.exr]\n"
    "         [--stats stats.json]\n"
    "         [-w 800] [-h 600] [--yaw 35] [--pitch 20]\n"
    "         [--sun-yaw 45] [--sun-elev 50] [--turbidity 2.5] [--exposure 0.0]\n",
    argv0);
}

bool loadTextureSRGB(const char* path, std::vector<float>& out, int& w, int& h) {
  int ch;
  uint8_t* data = stbi_load(path, &w, &h, &ch, 3);
  if (!data) {
    fprintf(stderr, "stb_image: failed to load %s\n", path);
    return false;
  }
  out.resize((size_t)w * h * 3);
  // sRGB -> linear.
  for (size_t i = 0; i < (size_t)w * h * 3; ++i) {
    float x = data[i] / 255.0f;
    out[i] = (x <= 0.04045f) ? x / 12.92f
                             : std::pow((x + 0.055f) / 1.055f, 2.4f);
  }
  stbi_image_free(data);
  return true;
}

int writeBeautyPNG(const char* path, const trellis2::AOVImage& img, float exposure) {
  std::vector<uint8_t> ldr;
  trellis2::encodeBeautyLDR(img, ldr, exposure, /*srgb=*/true);
  int ok = stbi_write_png(path, (int)img.width, (int)img.height, 3,
                          ldr.data(), (int)img.width * 3);
  if (!ok) {
    fprintf(stderr, "stb_image_write: failed %s\n", path);
    return -1;
  }
  fprintf(stderr, "wrote %s (%ux%u, sRGB)\n", path, img.width, img.height);
  return 0;
}

// EXR channel pull-out for one float per pixel: image is [N,3] interleaved.
std::vector<float> sliceChan(const std::vector<float>& src, size_t n, int comp, int stride) {
  std::vector<float> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = src[i * stride + comp];
  return out;
}

int writeAOVEXR(const char* path, const trellis2::AOVImage& img) {
  size_t N = (size_t)img.width * img.height;
  // Channels (alphabetical for tinyexr):
  //   A (mask), B/G/R (beauty), albedoB/G/R, depth, normalX/Y/Z, posX/Y/Z
  // We'll just be explicit and order them ourselves; tinyexr writes whatever order we give.
  struct Ch { const char* name; std::vector<float> data; };
  std::vector<Ch> chans;
  // Beauty as B,G,R (so naive viewers show it as "default")
  chans.push_back({"R",        sliceChan(img.beauty,   N, 0, 3)});
  chans.push_back({"G",        sliceChan(img.beauty,   N, 1, 3)});
  chans.push_back({"B",        sliceChan(img.beauty,   N, 2, 3)});
  chans.push_back({"A",        img.mask});
  chans.push_back({"albedo.R", sliceChan(img.albedo,   N, 0, 3)});
  chans.push_back({"albedo.G", sliceChan(img.albedo,   N, 1, 3)});
  chans.push_back({"albedo.B", sliceChan(img.albedo,   N, 2, 3)});
  chans.push_back({"normal.X", sliceChan(img.normal,   N, 0, 3)});
  chans.push_back({"normal.Y", sliceChan(img.normal,   N, 1, 3)});
  chans.push_back({"normal.Z", sliceChan(img.normal,   N, 2, 3)});
  chans.push_back({"P.X",      sliceChan(img.position, N, 0, 3)});
  chans.push_back({"P.Y",      sliceChan(img.position, N, 1, 3)});
  chans.push_back({"P.Z",      sliceChan(img.position, N, 2, 3)});
  chans.push_back({"Z",        img.depth});
  // face_id as float (-1 on miss). Lets image_diff reconstruct regions.
  std::vector<float> fid_f((size_t)img.width * img.height);
  for (size_t i = 0; i < fid_f.size(); ++i) fid_f[i] = (float)img.face_id[i];
  chans.push_back({"face_id",  fid_f});

  EXRHeader header; InitEXRHeader(&header);
  EXRImage  image;  InitEXRImage(&image);
  int n = (int)chans.size();
  std::vector<float*> ptrs(n);
  for (int i = 0; i < n; ++i) ptrs[i] = chans[i].data.data();
  image.num_channels = n;
  image.width = (int)img.width;
  image.height = (int)img.height;
  image.images = (unsigned char**)ptrs.data();

  header.num_channels = n;
  header.channels = (EXRChannelInfo*)malloc(sizeof(EXRChannelInfo) * (size_t)n);
  header.pixel_types = (int*)malloc(sizeof(int) * (size_t)n);
  header.requested_pixel_types = (int*)malloc(sizeof(int) * (size_t)n);
  for (int i = 0; i < n; ++i) {
    strncpy(header.channels[i].name, chans[i].name, 255);
    header.channels[i].name[255] = '\0';
    header.pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
  }
  const char* err = NULL;
  int ret = SaveEXRImageToFile(&image, &header, path, &err);
  free(header.channels);
  free(header.pixel_types);
  free(header.requested_pixel_types);
  if (ret != TINYEXR_SUCCESS) {
    fprintf(stderr, "EXR save failed: %s\n", err ? err : "unknown");
    FreeEXRErrorMessage(err);
    return -1;
  }
  fprintf(stderr, "wrote %s (%dx%d, %d channels)\n", path, image.width, image.height, n);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const char* in_path  = nullptr;
  const char* tex_path = nullptr;
  const char* png_path = nullptr;
  const char* exr_path = nullptr;
  const char* stats_path = nullptr;
  uint32_t W = 800, H = 600;
  float yaw = 35.0f, pitch = 20.0f;
  float sun_yaw = 45.0f, sun_elev = 50.0f;
  float turbidity = 2.5f;
  float exposure = 0.0f;

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&](const char* opt) -> const char* {
      if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", opt); std::exit(2); }
      return argv[++i];
    };
    if      (a == "--in")        in_path = need("--in");
    else if (a == "--texture")   tex_path = need("--texture");
    else if (a == "--out")       png_path = need("--out");
    else if (a == "--exr")       exr_path = need("--exr");
    else if (a == "--stats")     stats_path = need("--stats");
    else if (a == "-w")          W = (uint32_t)std::atoi(need("-w"));
    else if (a == "-h")          H = (uint32_t)std::atoi(need("-h"));
    else if (a == "--yaw")       yaw = (float)std::atof(need("--yaw"));
    else if (a == "--pitch")     pitch = (float)std::atof(need("--pitch"));
    else if (a == "--sun-yaw")   sun_yaw = (float)std::atof(need("--sun-yaw"));
    else if (a == "--sun-elev")  sun_elev = (float)std::atof(need("--sun-elev"));
    else if (a == "--turbidity") turbidity = (float)std::atof(need("--turbidity"));
    else if (a == "--exposure")  exposure = (float)std::atof(need("--exposure"));
    else { usage(argv[0]); return 2; }
  }
  if (!in_path) { usage(argv[0]); return 2; }
  if (!png_path && !exr_path) png_path = "preview.png";

  // ---- load mesh ---------------------------------------------------------
  trellis2::PreviewScene scene;
  trellis2::Mesh m;
  if (!trellis2::loadMesh(in_path, m)) return 1;
  scene.vertices = m.v;
  scene.faces    = m.f;
  // Optional vertex_normals/uvs from the same file.
  st_context* ctx = safetensors_open(in_path);
  if (ctx) {
    int in_n = safetensors_find(ctx, "vertex_normals");
    if (in_n >= 0 && std::strcmp(safetensors_dtype(ctx, in_n), "F32") == 0) {
      size_t nb = safetensors_nbytes(ctx, in_n);
      const float* p = (const float*)safetensors_data(ctx, in_n);
      scene.vertex_normals.assign(p, p + nb / sizeof(float));
    }
    int in_uv = safetensors_find(ctx, "uvs");
    if (in_uv >= 0 && std::strcmp(safetensors_dtype(ctx, in_uv), "F32") == 0) {
      size_t nb = safetensors_nbytes(ctx, in_uv);
      const float* p = (const float*)safetensors_data(ctx, in_uv);
      scene.uvs.assign(p, p + nb / sizeof(float));
    }
    int in_mid = safetensors_find(ctx, "face_material_id");
    if (in_mid >= 0 && std::strcmp(safetensors_dtype(ctx, in_mid), "I32") == 0) {
      size_t nb = safetensors_nbytes(ctx, in_mid);
      const int32_t* p = (const int32_t*)safetensors_data(ctx, in_mid);
      scene.face_material_id.assign(p, p + nb / sizeof(int32_t));
    }
    safetensors_close(ctx);
  }
  fprintf(stderr, "mesh: V=%u F=%u (uvs=%s)\n",
          (uint32_t)(scene.vertices.size()/3), (uint32_t)(scene.faces.size()/3),
          scene.uvs.empty() ? "no" : "yes");

  // ---- material + texture ------------------------------------------------
  scene.material.base_color = lightrt::Vec3(0.8f, 0.8f, 0.8f);
  scene.material.metallic = 0.0f;
  scene.material.roughness = 0.55f;
  if (tex_path) {
    if (!loadTextureSRGB(tex_path, scene.material.tex,
                         scene.material.tex_w, scene.material.tex_h)) {
      return 1;
    }
    fprintf(stderr, "texture: %s (%dx%d)\n", tex_path,
            scene.material.tex_w, scene.material.tex_h);
  }

  // ---- sun+sky -----------------------------------------------------------
  {
    float sy = sun_yaw * (float)M_PI / 180.0f;
    float se = sun_elev * (float)M_PI / 180.0f;
    scene.sky.sun_dir = trellis2::vnorm(lightrt::Vec3(
        std::cos(se) * std::sin(sy),
        std::sin(se),
        std::cos(se) * std::cos(sy)));
    scene.sky.turbidity = turbidity;
  }

  // ---- build BVH ---------------------------------------------------------
  if (!scene.build()) {
    fprintf(stderr, "BVH build failed\n");
    return 1;
  }

  // ---- frame the camera around the mesh AABB -----------------------------
  lightrt::Vec3 mn(1e30f, 1e30f, 1e30f), mx(-1e30f, -1e30f, -1e30f);
  for (size_t i = 0; i < scene.vertices.size(); i += 3) {
    mn.x = std::min(mn.x, scene.vertices[i+0]); mx.x = std::max(mx.x, scene.vertices[i+0]);
    mn.y = std::min(mn.y, scene.vertices[i+1]); mx.y = std::max(mx.y, scene.vertices[i+1]);
    mn.z = std::min(mn.z, scene.vertices[i+2]); mx.z = std::max(mx.z, scene.vertices[i+2]);
  }
  lightrt::Vec3 ctr = (mn + mx) * 0.5f;
  lightrt::Vec3 ext = mx - mn;
  float radius = trellis2::vlen(ext) * 0.85f;
  trellis2::Camera cam = trellis2::Camera::fromOrbit(ctr, radius, yaw, pitch, 35.0f);

  // ---- render ------------------------------------------------------------
  trellis2::AOVImage img;
  img.allocate(W, H);
  trellis2::render(scene, cam, img);

  // ---- write -------------------------------------------------------------
  int rc = 0;
  if (png_path) rc |= writeBeautyPNG(png_path, img, exposure);
  if (exr_path) rc |= writeAOVEXR(exr_path, img);
  if (stats_path) {
    trellis2::StatsOpts opts;
    trellis2::StatsReport rep = trellis2::summarize(img, scene.face_material_id, opts);
    if (!trellis2::writeStatsJSON(rep, stats_path)) rc |= 1;
    else fprintf(stderr, "wrote %s (hit_fraction=%.3f, regions=%zu)\n",
                 stats_path, rep.hit_fraction, rep.regions.size());
  }
  return rc;
}
