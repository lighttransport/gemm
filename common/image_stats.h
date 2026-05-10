// LLM-friendly numeric summaries of trellis2::AOVImage renders.
//
// Header-only. Two entry points:
//   summarize(img, face_id, material_id, opts) -> StatsReport
//   diff(a, b, face_id_a, material_id_a, opts) -> DiffReport
// Plus writeStatsJSON / writeDiffJSON for compact JSON output.
//
// Designed so an LLM agent can compare two renders by reading a few hundred
// bytes of JSON instead of invoking a vision model.

#pragma once

#include "preview_render.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace trellis2 {

struct StatsOpts {
  int   hist_bins = 32;
  float lum_max   = 4.0f;
  float depth_max_quantile = 0.99f;
  std::vector<std::pair<int, std::string>> id_names;
  // For diff: pixel is "changed" when |delta_lum| > changed_eps.
  float changed_eps = 1e-3f;
};

struct ChannelStats {
  float min = 0, max = 0, mean = 0, std = 0;
  float p1 = 0, p10 = 0, p50 = 0, p90 = 0, p99 = 0;
  std::vector<uint32_t> histogram;
  float hist_min = 0, hist_max = 0;
};

struct RegionStats {
  int id = 0;
  std::string name;
  uint32_t pixel_count = 0;
  float area_fraction = 0;
  int bbox[4] = {0,0,0,0};   // x0, y0, x1, y1 inclusive
  float color_mean[3] = {0,0,0};
  float depth_mean = 0;
  float depth_range[2] = {0,0};
  float normal_mean[3] = {0,0,0};
};

struct StatsReport {
  uint32_t width = 0, height = 0;
  uint32_t hit_pixels = 0;
  float    hit_fraction = 0;
  ChannelStats luminance, R, G, B, depth, normal_y;
  std::vector<RegionStats> regions;
};

struct DiffReport {
  uint32_t width = 0, height = 0;
  float psnr_db = 0, mae = 0, max_abs_diff = 0;
  float changed_fraction = 0;
  ChannelStats delta_R, delta_G, delta_B;
  float mask_iou = 0;
  float depth_mae = 0, depth_psnr_db = 0;
  float normal_cosine_mean = 0;
  std::vector<RegionStats> region_delta;
};

// ---- internals ------------------------------------------------------------

namespace stats_detail {

inline float lum709(float r, float g, float b) {
  return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}

// Computes basic stats and a uniform histogram. `hist_min/hist_max` are the
// histogram domain (auto if hist_max <= hist_min).
inline ChannelStats summarizeChannel(const std::vector<float>& vals,
                                     int bins,
                                     float forced_min = 0.0f,
                                     float forced_max = -1.0f) {
  ChannelStats s;
  s.histogram.assign((size_t)std::max(1, bins), 0u);
  if (vals.empty()) return s;

  float vmin = vals[0], vmax = vals[0];
  double sum = 0.0;
  for (float v : vals) { vmin = std::min(vmin, v); vmax = std::max(vmax, v); sum += v; }
  s.min = vmin; s.max = vmax;
  s.mean = (float)(sum / vals.size());
  double var = 0.0;
  for (float v : vals) { double d = v - s.mean; var += d * d; }
  s.std = (float)std::sqrt(var / std::max<size_t>(1, vals.size()));

  // Quantiles via partial sort copy.
  std::vector<float> tmp = vals;
  auto q = [&](float p) {
    if (tmp.empty()) return 0.0f;
    size_t k = (size_t)std::min((float)tmp.size() - 1.0f,
                                std::max(0.0f, p * (tmp.size() - 1)));
    std::nth_element(tmp.begin(), tmp.begin() + k, tmp.end());
    return tmp[k];
  };
  s.p1  = q(0.01f);
  s.p10 = q(0.10f);
  s.p50 = q(0.50f);
  s.p90 = q(0.90f);
  s.p99 = q(0.99f);

  float hmin = forced_min, hmax = forced_max;
  if (hmax <= hmin) { hmin = vmin; hmax = vmax; }
  if (hmax <= hmin) hmax = hmin + 1.0f;
  s.hist_min = hmin; s.hist_max = hmax;
  float scale = (float)bins / (hmax - hmin);
  for (float v : vals) {
    int b = (int)((v - hmin) * scale);
    if (b < 0) b = 0; else if (b >= bins) b = bins - 1;
    s.histogram[(size_t)b]++;
  }
  return s;
}

// Build region map from face_id -> material_id (or face_id directly).
inline std::vector<int32_t> derive_pixel_id(
    const std::vector<int32_t>& face_id_pixels,
    const std::vector<int32_t>& face_material_id) {
  std::vector<int32_t> out(face_id_pixels.size(), -1);
  if (face_id_pixels.empty()) return out;
  bool have_mat = !face_material_id.empty();
  for (size_t i = 0; i < face_id_pixels.size(); ++i) {
    int32_t f = face_id_pixels[i];
    if (f < 0) { out[i] = -1; continue; }
    if (have_mat && (size_t)f < face_material_id.size())
      out[i] = face_material_id[(size_t)f];
    else
      out[i] = -1;  // no material map -> single-region "0" handled below
  }
  return out;
}

inline std::vector<RegionStats> compute_regions(
    const AOVImage& img,
    const std::vector<int32_t>& pixel_id,
    const StatsOpts& opts) {
  std::vector<RegionStats> out;
  if (img.width == 0 || img.height == 0) return out;
  size_t N = (size_t)img.width * img.height;

  // If no per-pixel material id, treat all hits as one "default" region.
  bool any_id = false;
  for (size_t i = 0; i < N && i < pixel_id.size(); ++i) {
    if (pixel_id[i] >= 0) { any_id = true; break; }
  }

  std::unordered_map<int, RegionStats> acc;
  std::unordered_map<int, std::array<double, 3>> col_sum;
  std::unordered_map<int, double>    depth_sum;
  std::unordered_map<int, std::pair<float, float>> depth_minmax;
  std::unordered_map<int, std::array<double, 3>> norm_sum;

  for (size_t i = 0; i < N; ++i) {
    if (img.mask[i] <= 0.0f) continue;
    int id;
    if (any_id) {
      if (i >= pixel_id.size() || pixel_id[i] < 0) continue;
      id = pixel_id[i];
    } else {
      id = 0;
    }
    int x = (int)(i % img.width);
    int y = (int)(i / img.width);
    auto& r = acc[id];
    if (r.pixel_count == 0) {
      r.id = id;
      r.bbox[0] = x; r.bbox[1] = y; r.bbox[2] = x; r.bbox[3] = y;
      depth_minmax[id] = {img.depth[i], img.depth[i]};
    } else {
      r.bbox[0] = std::min(r.bbox[0], x);
      r.bbox[1] = std::min(r.bbox[1], y);
      r.bbox[2] = std::max(r.bbox[2], x);
      r.bbox[3] = std::max(r.bbox[3], y);
      auto& dm = depth_minmax[id];
      dm.first  = std::min(dm.first,  img.depth[i]);
      dm.second = std::max(dm.second, img.depth[i]);
    }
    r.pixel_count++;
    auto& cs = col_sum[id];
    cs[0] += img.beauty[3*i+0];
    cs[1] += img.beauty[3*i+1];
    cs[2] += img.beauty[3*i+2];
    auto& ns = norm_sum[id];
    ns[0] += img.normal[3*i+0];
    ns[1] += img.normal[3*i+1];
    ns[2] += img.normal[3*i+2];
    depth_sum[id] += img.depth[i];
  }

  std::unordered_map<int, std::string> name_map;
  for (auto& kv : opts.id_names) name_map[kv.first] = kv.second;

  out.reserve(acc.size());
  for (auto& kv : acc) {
    RegionStats r = kv.second;
    int id = r.id;
    auto it = name_map.find(id);
    char fallback[32];
    if (it != name_map.end()) r.name = it->second;
    else { std::snprintf(fallback, sizeof(fallback), "id_%d", id); r.name = fallback; }
    float n = (float)r.pixel_count;
    r.area_fraction = n / (float)N;
    auto& cs = col_sum[id];
    r.color_mean[0] = (float)(cs[0] / n);
    r.color_mean[1] = (float)(cs[1] / n);
    r.color_mean[2] = (float)(cs[2] / n);
    auto& ns = norm_sum[id];
    r.normal_mean[0] = (float)(ns[0] / n);
    r.normal_mean[1] = (float)(ns[1] / n);
    r.normal_mean[2] = (float)(ns[2] / n);
    r.depth_mean = (float)(depth_sum[id] / n);
    auto& dm = depth_minmax[id];
    r.depth_range[0] = dm.first;
    r.depth_range[1] = dm.second;
    out.push_back(r);
  }
  std::sort(out.begin(), out.end(),
            [](const RegionStats& a, const RegionStats& b){ return a.id < b.id; });
  return out;
}

}  // namespace stats_detail

// ---- public API -----------------------------------------------------------

inline StatsReport summarize(const AOVImage& img,
                             const std::vector<int32_t>& face_material_id,
                             const StatsOpts& opts) {
  using namespace stats_detail;
  StatsReport r;
  r.width = img.width; r.height = img.height;
  size_t N = (size_t)img.width * img.height;
  if (N == 0) return r;

  // Hit mask scalars.
  std::vector<float> R, G, B, lum, depth_hits, ny_hits;
  R.reserve(N); G.reserve(N); B.reserve(N); lum.reserve(N);
  depth_hits.reserve(N); ny_hits.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    if (img.mask[i] <= 0.0f) continue;
    float r_ = img.beauty[3*i+0], g_ = img.beauty[3*i+1], b_ = img.beauty[3*i+2];
    R.push_back(r_); G.push_back(g_); B.push_back(b_);
    lum.push_back(lum709(r_, g_, b_));
    depth_hits.push_back(img.depth[i]);
    ny_hits.push_back(img.normal[3*i+1]);
  }
  r.hit_pixels = (uint32_t)R.size();
  r.hit_fraction = (float)R.size() / (float)N;

  r.R         = summarizeChannel(R,   opts.hist_bins, 0.0f, opts.lum_max);
  r.G         = summarizeChannel(G,   opts.hist_bins, 0.0f, opts.lum_max);
  r.B         = summarizeChannel(B,   opts.hist_bins, 0.0f, opts.lum_max);
  r.luminance = summarizeChannel(lum, opts.hist_bins, 0.0f, opts.lum_max);
  r.depth     = summarizeChannel(depth_hits, opts.hist_bins);
  r.normal_y  = summarizeChannel(ny_hits, opts.hist_bins, -1.0f, 1.0f);

  // Region stats use face_id channel + optional per-face material map.
  auto pixel_id = derive_pixel_id(img.face_id, face_material_id);
  r.regions = compute_regions(img, pixel_id, opts);
  return r;
}

inline DiffReport diff(const AOVImage& a, const AOVImage& b,
                       const std::vector<int32_t>& face_material_id_a,
                       const StatsOpts& opts) {
  using namespace stats_detail;
  DiffReport d;
  d.width = a.width; d.height = a.height;
  if (a.width != b.width || a.height != b.height) {
    std::fprintf(stderr, "diff: size mismatch %ux%u vs %ux%u\n",
                 a.width, a.height, b.width, b.height);
    return d;
  }
  size_t N = (size_t)a.width * a.height;
  if (N == 0) return d;

  std::vector<float> dR, dG, dB;
  dR.reserve(N); dG.reserve(N); dB.reserve(N);
  double sse = 0.0, mae = 0.0;
  float maxabs = 0.0f;
  uint32_t changed = 0;
  uint32_t inter = 0, uni = 0;
  double depth_sse = 0.0, depth_mae = 0.0;
  uint32_t depth_n = 0;
  double cos_sum = 0.0;
  uint32_t cos_n = 0;

  for (size_t i = 0; i < N; ++i) {
    float dr = a.beauty[3*i+0] - b.beauty[3*i+0];
    float dg = a.beauty[3*i+1] - b.beauty[3*i+1];
    float db = a.beauty[3*i+2] - b.beauty[3*i+2];
    dR.push_back(dr); dG.push_back(dg); dB.push_back(db);
    sse += (double)(dr*dr + dg*dg + db*db);
    mae += (double)(std::fabs(dr) + std::fabs(dg) + std::fabs(db));
    maxabs = std::max(maxabs, std::max({std::fabs(dr), std::fabs(dg), std::fabs(db)}));
    float dl = lum709(dr, dg, db);
    if (std::fabs(dl) > opts.changed_eps) changed++;

    bool ma = a.mask[i] > 0.0f, mb = b.mask[i] > 0.0f;
    if (ma || mb) uni++;
    if (ma && mb) {
      inter++;
      float dz = a.depth[i] - b.depth[i];
      depth_sse += dz * dz;
      depth_mae += std::fabs(dz);
      depth_n++;
      lightrt::Vec3 na(a.normal[3*i+0], a.normal[3*i+1], a.normal[3*i+2]);
      lightrt::Vec3 nb(b.normal[3*i+0], b.normal[3*i+1], b.normal[3*i+2]);
      float la = std::sqrt(na.dot(na)), lb = std::sqrt(nb.dot(nb));
      if (la > 1e-6f && lb > 1e-6f) {
        cos_sum += (double)(na.dot(nb) / (la * lb));
        cos_n++;
      }
    }
  }

  d.mae = (float)(mae / (3.0 * N));
  float mse = (float)(sse / (3.0 * N));
  d.psnr_db = mse > 0 ? (float)(10.0 * std::log10(1.0 / mse)) : 99.0f;
  d.max_abs_diff = maxabs;
  d.changed_fraction = (float)changed / (float)N;
  d.delta_R = summarizeChannel(dR, opts.hist_bins);
  d.delta_G = summarizeChannel(dG, opts.hist_bins);
  d.delta_B = summarizeChannel(dB, opts.hist_bins);
  d.mask_iou = uni ? (float)inter / (float)uni : 1.0f;
  d.depth_mae = depth_n ? (float)(depth_mae / depth_n) : 0.0f;
  float depth_mse = depth_n ? (float)(depth_sse / depth_n) : 0.0f;
  d.depth_psnr_db = depth_mse > 0 ? (float)(10.0 * std::log10(1.0 / depth_mse)) : 99.0f;
  d.normal_cosine_mean = cos_n ? (float)(cos_sum / cos_n) : 1.0f;

  // Region delta: sum over A's region pixels that are also hit in B.
  auto pid_a = derive_pixel_id(a.face_id, face_material_id_a);
  bool any_id = false;
  for (auto v : pid_a) if (v >= 0) { any_id = true; break; }
  std::unordered_map<int, RegionStats> acc;
  std::unordered_map<int, std::array<double, 3>> col_sum;
  std::unordered_map<int, double>    depth_sum;
  for (size_t i = 0; i < N; ++i) {
    if (a.mask[i] <= 0.0f || b.mask[i] <= 0.0f) continue;
    int id;
    if (any_id) {
      if (i >= pid_a.size() || pid_a[i] < 0) continue;
      id = pid_a[i];
    } else id = 0;
    auto& r = acc[id];
    r.id = id;
    r.pixel_count++;
    auto& cs = col_sum[id];
    cs[0] += (a.beauty[3*i+0] - b.beauty[3*i+0]);
    cs[1] += (a.beauty[3*i+1] - b.beauty[3*i+1]);
    cs[2] += (a.beauty[3*i+2] - b.beauty[3*i+2]);
    depth_sum[id] += (a.depth[i] - b.depth[i]);
  }
  std::unordered_map<int, std::string> name_map;
  for (auto& kv : opts.id_names) name_map[kv.first] = kv.second;
  for (auto& kv : acc) {
    RegionStats r = kv.second;
    int id = r.id;
    float n = (float)r.pixel_count;
    auto it = name_map.find(id);
    char fallback[32];
    if (it != name_map.end()) r.name = it->second;
    else { std::snprintf(fallback, sizeof(fallback), "id_%d", id); r.name = fallback; }
    r.area_fraction = n / (float)N;
    auto& cs = col_sum[id];
    r.color_mean[0] = (float)(cs[0] / n);
    r.color_mean[1] = (float)(cs[1] / n);
    r.color_mean[2] = (float)(cs[2] / n);
    r.depth_mean = (float)(depth_sum[id] / n);
    d.region_delta.push_back(r);
  }
  std::sort(d.region_delta.begin(), d.region_delta.end(),
            [](const RegionStats& a, const RegionStats& b){ return a.id < b.id; });
  return d;
}

// ---- JSON writers (compact, no external lib) ------------------------------

namespace stats_detail {

inline void emit_chan(FILE* fp, const char* name, const ChannelStats& c) {
  std::fprintf(fp,
    "\"%s\":{\"min\":%.6g,\"max\":%.6g,\"mean\":%.6g,\"std\":%.6g,"
    "\"p1\":%.6g,\"p10\":%.6g,\"p50\":%.6g,\"p90\":%.6g,\"p99\":%.6g,"
    "\"hist_min\":%.6g,\"hist_max\":%.6g,\"hist\":[",
    name, c.min, c.max, c.mean, c.std,
    c.p1, c.p10, c.p50, c.p90, c.p99, c.hist_min, c.hist_max);
  for (size_t i = 0; i < c.histogram.size(); ++i) {
    std::fprintf(fp, "%s%u", i ? "," : "", c.histogram[i]);
  }
  std::fputs("]}", fp);
}

inline void emit_region(FILE* fp, const RegionStats& r) {
  std::fprintf(fp,
    "{\"id\":%d,\"name\":\"%s\",\"pixels\":%u,\"area\":%.6g,"
    "\"bbox\":[%d,%d,%d,%d],"
    "\"color_mean\":[%.6g,%.6g,%.6g],"
    "\"depth_mean\":%.6g,\"depth_range\":[%.6g,%.6g],"
    "\"normal_mean\":[%.6g,%.6g,%.6g]}",
    r.id, r.name.c_str(), r.pixel_count, r.area_fraction,
    r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3],
    r.color_mean[0], r.color_mean[1], r.color_mean[2],
    r.depth_mean, r.depth_range[0], r.depth_range[1],
    r.normal_mean[0], r.normal_mean[1], r.normal_mean[2]);
}

}  // namespace stats_detail

inline bool writeStatsJSON(const StatsReport& r, const char* path) {
  FILE* fp = std::fopen(path, "wb");
  if (!fp) { std::fprintf(stderr, "writeStatsJSON: cannot open %s\n", path); return false; }
  std::fprintf(fp, "{\"size\":[%u,%u],\"hit_pixels\":%u,\"hit_fraction\":%.6g,",
               r.width, r.height, r.hit_pixels, r.hit_fraction);
  stats_detail::emit_chan(fp, "luminance", r.luminance); std::fputc(',', fp);
  stats_detail::emit_chan(fp, "R",         r.R);         std::fputc(',', fp);
  stats_detail::emit_chan(fp, "G",         r.G);         std::fputc(',', fp);
  stats_detail::emit_chan(fp, "B",         r.B);         std::fputc(',', fp);
  stats_detail::emit_chan(fp, "depth",     r.depth);     std::fputc(',', fp);
  stats_detail::emit_chan(fp, "normal_y",  r.normal_y);
  std::fputs(",\"regions\":[", fp);
  for (size_t i = 0; i < r.regions.size(); ++i) {
    if (i) std::fputc(',', fp);
    stats_detail::emit_region(fp, r.regions[i]);
  }
  std::fputs("]}\n", fp);
  std::fclose(fp);
  return true;
}

inline bool writeDiffJSON(const DiffReport& d, const char* path) {
  FILE* fp = std::fopen(path, "wb");
  if (!fp) { std::fprintf(stderr, "writeDiffJSON: cannot open %s\n", path); return false; }
  std::fprintf(fp,
    "{\"size\":[%u,%u],\"psnr_db\":%.6g,\"mae\":%.6g,\"max_abs_diff\":%.6g,"
    "\"changed_fraction\":%.6g,\"mask_iou\":%.6g,\"depth_mae\":%.6g,"
    "\"depth_psnr_db\":%.6g,\"normal_cosine_mean\":%.6g,",
    d.width, d.height, d.psnr_db, d.mae, d.max_abs_diff,
    d.changed_fraction, d.mask_iou, d.depth_mae, d.depth_psnr_db,
    d.normal_cosine_mean);
  stats_detail::emit_chan(fp, "delta_R", d.delta_R); std::fputc(',', fp);
  stats_detail::emit_chan(fp, "delta_G", d.delta_G); std::fputc(',', fp);
  stats_detail::emit_chan(fp, "delta_B", d.delta_B);
  std::fputs(",\"regions\":[", fp);
  for (size_t i = 0; i < d.region_delta.size(); ++i) {
    if (i) std::fputc(',', fp);
    stats_detail::emit_region(fp, d.region_delta[i]);
  }
  std::fputs("]}\n", fp);
  std::fclose(fp);
  return true;
}

}  // namespace trellis2
