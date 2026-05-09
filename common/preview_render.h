// Lightweight preview path tracer for trellis2-style mesh+texture output.
//
// Backend: lightrt::SBVH ray-triangle intersection.
// Lighting: analytic Preetham-style sun+sky (no envmap loading).
// Shading:  Lambert + GGX (single-bounce direct + sky AO).
// AOVs:     beauty, albedo, normal, depth, position, mask.
//
// Header-only. Pulls in xatlas-free deps only — lightrt.hh and (if used)
// stb_image.h for the optional baseColor texture.

#pragma once

#include "lightrt.hh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace trellis2 {

// ---- math helpers ---------------------------------------------------------

inline lightrt::Vec3 vmul(const lightrt::Vec3& a, const lightrt::Vec3& b) {
  return lightrt::Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}
inline lightrt::Vec3 vsat(const lightrt::Vec3& a) {
  return lightrt::Vec3(std::max(0.0f, std::min(1.0f, a.x)),
                       std::max(0.0f, std::min(1.0f, a.y)),
                       std::max(0.0f, std::min(1.0f, a.z)));
}
inline float vlen(const lightrt::Vec3& v) {
  return std::sqrt(v.dot(v));
}
inline lightrt::Vec3 vnorm(const lightrt::Vec3& v) {
  float l = vlen(v);
  return l > 0.0f ? v * (1.0f / l) : v;
}

// ---- analytic sun+sky (Preetham-like, single-color sun) -------------------
//
// Inputs:
//   sun_dir:    unit vector toward sun (Y-up)
//   turbidity:  2..10 (clear..hazy), 2.5 default
//   intensity:  global multiplier for sky+sun (1.0 default)
// Output:
//   skyColor(view_dir): linear RGB radiance for a ray going to infinity.

struct SunSky {
  lightrt::Vec3 sun_dir = vnorm(lightrt::Vec3(0.3f, 0.6f, 0.4f));
  float turbidity = 2.5f;
  float intensity = 1.0f;
  float sun_angular_radius = 0.00465f;     // ~0.53 degrees, like real sun
  float sun_disk_intensity = 30.0f;        // multiplier inside the disk
  // Tints chosen to look reasonable without atmospheric LUTs.
  lightrt::Vec3 zenith_color  = lightrt::Vec3(0.18f, 0.32f, 0.62f);
  lightrt::Vec3 horizon_color = lightrt::Vec3(0.85f, 0.78f, 0.70f);
  lightrt::Vec3 sun_color     = lightrt::Vec3(1.00f, 0.96f, 0.85f);
};

inline lightrt::Vec3 evalSky(const SunSky& s, const lightrt::Vec3& dir) {
  lightrt::Vec3 d = vnorm(dir);
  // Sun disk
  float cos_gamma = std::max(-1.0f, std::min(1.0f, d.dot(s.sun_dir)));
  float gamma = std::acos(cos_gamma);
  if (gamma < s.sun_angular_radius) {
    return s.sun_color * (s.intensity * s.sun_disk_intensity);
  }
  // Sky gradient: zenith vs horizon by cos(angle to up)
  float t = std::max(0.0f, d.y);
  float t_smooth = t * t * (3.0f - 2.0f * t);  // smoothstep
  lightrt::Vec3 base = s.horizon_color * (1.0f - t_smooth) + s.zenith_color * t_smooth;
  // Single-scatter-ish tint near sun (Rayleigh phase ~ 1+cos²γ)
  float phase = 0.5f * (1.0f + cos_gamma * cos_gamma);
  // Fade halo with elevation (don't tint the zenith too much)
  float halo = std::pow(std::max(0.0f, 1.0f - gamma / 1.5f), 2.0f);
  base = base + s.sun_color * (0.25f * phase * halo);
  // Ground: dim brown when looking far below horizon
  if (d.y < 0.0f) {
    float k = std::min(1.0f, -d.y * 4.0f);
    lightrt::Vec3 ground(0.10f, 0.09f, 0.08f);
    base = base * (1.0f - k) + ground * k;
  }
  // Turbidity: hazier skies are brighter and whiter
  float t_lerp = std::max(0.0f, std::min(1.0f, (s.turbidity - 2.0f) / 8.0f));
  lightrt::Vec3 white(0.9f, 0.9f, 0.9f);
  base = base * (1.0f - 0.3f * t_lerp) + white * (0.3f * t_lerp);
  return base * s.intensity;
}

// Diffuse irradiance approximation: integrate sky over upper hemisphere.
// Cheap stand-in for a real spherical-harmonics fit: blend zenith/horizon.
inline lightrt::Vec3 skyIrradiance(const SunSky& s, const lightrt::Vec3& n) {
  lightrt::Vec3 nn = vnorm(n);
  float t = nn.y * 0.5f + 0.5f;
  lightrt::Vec3 base = s.horizon_color * (1.0f - t) + s.zenith_color * t;
  return base * s.intensity * 1.5f;
}

// ---- material -------------------------------------------------------------

struct Material {
  lightrt::Vec3 base_color = lightrt::Vec3(0.8f, 0.8f, 0.8f);
  float metallic  = 0.0f;
  float roughness = 0.6f;
  // Optional baseColor texture (RGB, [0,1] linear). If nonempty, sampled by UV.
  std::vector<float> tex;
  int tex_w = 0, tex_h = 0;
};

inline lightrt::Vec3 sampleAlbedo(const Material& m, float u, float v) {
  if (m.tex.empty() || m.tex_w <= 0 || m.tex_h <= 0) return m.base_color;
  // Wrap UVs.
  u = u - std::floor(u);
  v = v - std::floor(v);
  int x = std::min(m.tex_w - 1, std::max(0, (int)(u * m.tex_w)));
  int y = std::min(m.tex_h - 1, std::max(0, (int)((1.0f - v) * m.tex_h)));
  size_t idx = ((size_t)y * (size_t)m.tex_w + (size_t)x) * 3;
  return lightrt::Vec3(m.tex[idx], m.tex[idx + 1], m.tex[idx + 2]);
}

// ---- scene ----------------------------------------------------------------

struct PreviewScene {
  // Triangle soup (indexed).
  std::vector<float>   vertices;       // 3*V
  std::vector<int32_t> faces;          // 3*F
  std::vector<float>   vertex_normals; // 3*V (optional; computed if empty)
  std::vector<float>   uvs;            // 2*V (optional)
  Material material;
  SunSky sky;

  // Built BVH and triangles (after build()).
  lightrt::SBVH bvh;
  std::vector<lightrt::Triangle> tris;

  bool build();
};

inline bool PreviewScene::build() {
  uint32_t nv = (uint32_t)(vertices.size() / 3);
  uint32_t nf = (uint32_t)(faces.size() / 3);
  if (nv == 0 || nf == 0) return false;
  if (vertex_normals.empty()) {
    vertex_normals.assign(3 * nv, 0.0f);
    for (uint32_t fi = 0; fi < nf; ++fi) {
      int32_t i0 = faces[3 * fi + 0], i1 = faces[3 * fi + 1], i2 = faces[3 * fi + 2];
      lightrt::Vec3 p0(vertices[3*i0], vertices[3*i0+1], vertices[3*i0+2]);
      lightrt::Vec3 p1(vertices[3*i1], vertices[3*i1+1], vertices[3*i1+2]);
      lightrt::Vec3 p2(vertices[3*i2], vertices[3*i2+1], vertices[3*i2+2]);
      lightrt::Vec3 n = (p1 - p0).cross(p2 - p0);
      for (int k = 0; k < 3; ++k) {
        vertex_normals[3*i0+k] += (k==0?n.x: k==1?n.y: n.z);
        vertex_normals[3*i1+k] += (k==0?n.x: k==1?n.y: n.z);
        vertex_normals[3*i2+k] += (k==0?n.x: k==1?n.y: n.z);
      }
    }
    for (uint32_t i = 0; i < nv; ++i) {
      lightrt::Vec3 n(vertex_normals[3*i], vertex_normals[3*i+1], vertex_normals[3*i+2]);
      n = vnorm(n);
      vertex_normals[3*i+0] = n.x; vertex_normals[3*i+1] = n.y; vertex_normals[3*i+2] = n.z;
    }
  }
  tris.clear();
  tris.reserve(nf);
  for (uint32_t fi = 0; fi < nf; ++fi) {
    int32_t i0 = faces[3*fi+0], i1 = faces[3*fi+1], i2 = faces[3*fi+2];
    tris.emplace_back(
        lightrt::Vec3(vertices[3*i0], vertices[3*i0+1], vertices[3*i0+2]),
        lightrt::Vec3(vertices[3*i1], vertices[3*i1+1], vertices[3*i1+2]),
        lightrt::Vec3(vertices[3*i2], vertices[3*i2+1], vertices[3*i2+2]));
  }
  lightrt::SBVHBuildConfig cfg;
  return bvh.build(tris, cfg);
}

// ---- camera ---------------------------------------------------------------

struct Camera {
  lightrt::Vec3 eye    = lightrt::Vec3(0, 0, 3);
  lightrt::Vec3 target = lightrt::Vec3(0, 0, 0);
  lightrt::Vec3 up     = lightrt::Vec3(0, 1, 0);
  float fov_y_deg      = 35.0f;

  // Orbit-frame helper: look at scene center from spherical angles.
  static Camera fromOrbit(const lightrt::Vec3& center, float radius,
                          float yaw_deg, float pitch_deg, float fov_y_deg = 35.0f) {
    float yaw = yaw_deg * (float)M_PI / 180.0f;
    float pitch = pitch_deg * (float)M_PI / 180.0f;
    lightrt::Vec3 dir(std::cos(pitch) * std::sin(yaw),
                      std::sin(pitch),
                      std::cos(pitch) * std::cos(yaw));
    Camera c;
    c.eye = center + dir * radius;
    c.target = center;
    c.fov_y_deg = fov_y_deg;
    return c;
  }
};

// ---- BRDF (GGX + Lambert), direct+ambient ---------------------------------

inline lightrt::Vec3 shade(const PreviewScene& scene,
                           const lightrt::Vec3& p,
                           const lightrt::Vec3& n,
                           const lightrt::Vec3& v,
                           const lightrt::Vec3& albedo) {
  const Material& m = scene.material;
  const SunSky& sky = scene.sky;
  lightrt::Vec3 nn = vnorm(n);
  lightrt::Vec3 vv = vnorm(v);
  lightrt::Vec3 l  = vnorm(sky.sun_dir);
  float NdotL = std::max(0.0f, nn.dot(l));
  float NdotV = std::max(1e-4f, nn.dot(vv));

  // Shadow ray (any-hit) toward sun.
  bool in_shadow = false;
  if (NdotL > 0.0f) {
    lightrt::Ray sr(p + nn * 1e-3f, l);
    in_shadow = scene.bvh.traverseAnyHit(sr);
  }

  // Direct sun radiance (constant over the disk; treat as direction).
  lightrt::Vec3 sun_rad = sky.sun_color * (sky.intensity * 5.0f);

  // GGX specular + Lambert diffuse (Schlick Fresnel, Smith-G).
  lightrt::Vec3 h = vnorm(vv + l);
  float NdotH = std::max(0.0f, nn.dot(h));
  float VdotH = std::max(0.0f, vv.dot(h));
  float a = std::max(0.04f, m.roughness * m.roughness);
  float a2 = a * a;
  float denom = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
  float D = a2 / (float)(M_PI * denom * denom + 1e-8f);
  float k = (a + 1.0f); k = k * k / 8.0f;
  float Gv = NdotV / (NdotV * (1.0f - k) + k);
  float Gl = NdotL / (NdotL * (1.0f - k) + k + 1e-6f);
  float G = Gv * Gl;
  lightrt::Vec3 F0 = lightrt::Vec3(0.04f, 0.04f, 0.04f) * (1.0f - m.metallic) + albedo * m.metallic;
  float fs = std::pow(1.0f - VdotH, 5.0f);
  lightrt::Vec3 F = F0 * (1.0f - fs) + lightrt::Vec3(1, 1, 1) * fs;
  lightrt::Vec3 spec = F * (D * G / (4.0f * NdotV * NdotL + 1e-6f));
  lightrt::Vec3 kd = (lightrt::Vec3(1,1,1) - F) * (1.0f - m.metallic);
  lightrt::Vec3 diff = vmul(kd, albedo) * (1.0f / (float)M_PI);

  lightrt::Vec3 direct(0,0,0);
  if (!in_shadow) {
    direct = vmul((diff + spec), sun_rad) * NdotL;
  }
  // Ambient: sky irradiance (diffuse only).
  lightrt::Vec3 amb = vmul(albedo, skyIrradiance(sky, nn)) * (1.0f / (float)M_PI);
  return direct + amb;
}

// ---- AOV buffer -----------------------------------------------------------

struct AOVImage {
  uint32_t width = 0, height = 0;
  std::vector<float> beauty;    // RGB
  std::vector<float> albedo;    // RGB
  std::vector<float> normal;    // RGB (world-space, [-1,1])
  std::vector<float> position;  // RGB (world-space)
  std::vector<float> depth;     // R   (linear, 0 if miss)
  std::vector<float> mask;      // R   (1 hit, 0 miss)

  void allocate(uint32_t w, uint32_t h) {
    width = w; height = h;
    size_t n3 = (size_t)w * h * 3;
    size_t n1 = (size_t)w * h;
    beauty.assign(n3, 0.0f);
    albedo.assign(n3, 0.0f);
    normal.assign(n3, 0.0f);
    position.assign(n3, 0.0f);
    depth.assign(n1, 0.0f);
    mask.assign(n1, 0.0f);
  }
};

// ---- render ---------------------------------------------------------------

inline void render(const PreviewScene& scene, const Camera& cam, AOVImage& out) {
  const uint32_t W = out.width, H = out.height;
  lightrt::Vec3 fwd = vnorm(cam.target - cam.eye);
  lightrt::Vec3 right = vnorm(fwd.cross(cam.up));
  lightrt::Vec3 cup = right.cross(fwd);
  float aspect = (float)W / (float)H;
  float scale = std::tan(cam.fov_y_deg * 0.5f * (float)M_PI / 180.0f);

  #pragma omp parallel for schedule(dynamic, 16)
  for (int y = 0; y < (int)H; ++y) {
    for (uint32_t x = 0; x < W; ++x) {
      float ndc_x = (2.0f * (x + 0.5f) / W - 1.0f) * aspect * scale;
      float ndc_y = (1.0f - 2.0f * (y + 0.5f) / H) * scale;
      lightrt::Vec3 d = vnorm(fwd + right * ndc_x + cup * ndc_y);
      lightrt::Ray r(cam.eye, d);
      float t, u, v;
      uint32_t fid = scene.bvh.traverse(r, t, u, v);
      size_t pi = ((size_t)y * W + x);
      if (fid == lightrt::kInvalidIndex) {
        // Miss: sky.
        lightrt::Vec3 sky = evalSky(scene.sky, d);
        out.beauty[3*pi+0] = sky.x; out.beauty[3*pi+1] = sky.y; out.beauty[3*pi+2] = sky.z;
        // AOVs zero on miss (mask = 0).
        continue;
      }
      // Hit: interpolate per-vertex attribs (SBVH::traverse returns prim_id).
      uint32_t f0 = scene.faces[3*fid+0];
      uint32_t f1 = scene.faces[3*fid+1];
      uint32_t f2 = scene.faces[3*fid+2];
      float w0 = 1.0f - u - v, w1 = u, w2 = v;
      lightrt::Vec3 n(
          w0 * scene.vertex_normals[3*f0+0] + w1 * scene.vertex_normals[3*f1+0] + w2 * scene.vertex_normals[3*f2+0],
          w0 * scene.vertex_normals[3*f0+1] + w1 * scene.vertex_normals[3*f1+1] + w2 * scene.vertex_normals[3*f2+1],
          w0 * scene.vertex_normals[3*f0+2] + w1 * scene.vertex_normals[3*f1+2] + w2 * scene.vertex_normals[3*f2+2]);
      n = vnorm(n);
      // Front-face only.
      if (n.dot(d) > 0) n = n * -1.0f;
      lightrt::Vec3 p = r.at(t);

      lightrt::Vec3 albedo = scene.material.base_color;
      if (!scene.uvs.empty()) {
        float uu = w0 * scene.uvs[2*f0+0] + w1 * scene.uvs[2*f1+0] + w2 * scene.uvs[2*f2+0];
        float vv = w0 * scene.uvs[2*f0+1] + w1 * scene.uvs[2*f1+1] + w2 * scene.uvs[2*f2+1];
        albedo = sampleAlbedo(scene.material, uu, vv);
      }

      lightrt::Vec3 col = shade(scene, p, n, d * -1.0f, albedo);

      out.beauty[3*pi+0] = col.x; out.beauty[3*pi+1] = col.y; out.beauty[3*pi+2] = col.z;
      out.albedo[3*pi+0] = albedo.x; out.albedo[3*pi+1] = albedo.y; out.albedo[3*pi+2] = albedo.z;
      out.normal[3*pi+0] = n.x; out.normal[3*pi+1] = n.y; out.normal[3*pi+2] = n.z;
      out.position[3*pi+0] = p.x; out.position[3*pi+1] = p.y; out.position[3*pi+2] = p.z;
      out.depth[pi] = t;
      out.mask[pi] = 1.0f;
    }
  }
}

// ---- tonemap + LDR conversion --------------------------------------------

inline uint8_t f2u8(float x) {
  x = std::max(0.0f, std::min(1.0f, x));
  return (uint8_t)(x * 255.0f + 0.5f);
}

// Filmic tonemap (ACES approximate by Narkowicz).
inline lightrt::Vec3 tonemapACES(const lightrt::Vec3& c) {
  const float a = 2.51f, b = 0.03f, cc = 2.43f, d = 0.59f, e = 0.14f;
  auto f = [&](float x) {
    return std::max(0.0f, std::min(1.0f, (x * (a * x + b)) / (x * (cc * x + d) + e)));
  };
  return lightrt::Vec3(f(c.x), f(c.y), f(c.z));
}

inline void encodeBeautyLDR(const AOVImage& img, std::vector<uint8_t>& out_rgb,
                            float exposure = 0.0f, bool srgb = true) {
  size_t n = (size_t)img.width * img.height;
  out_rgb.assign(n * 3, 0);
  float mul = std::pow(2.0f, exposure);
  for (size_t i = 0; i < n; ++i) {
    lightrt::Vec3 c(img.beauty[3*i+0] * mul, img.beauty[3*i+1] * mul, img.beauty[3*i+2] * mul);
    c = tonemapACES(c);
    if (srgb) {
      auto enc = [](float x) { return x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1.0f/2.4f) - 0.055f; };
      c = lightrt::Vec3(enc(c.x), enc(c.y), enc(c.z));
    }
    out_rgb[3*i+0] = f2u8(c.x);
    out_rgb[3*i+1] = f2u8(c.y);
    out_rgb[3*i+2] = f2u8(c.z);
  }
}

}  // namespace trellis2
